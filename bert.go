package bert

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// Token represents a token ID in the model's vocabulary.
type Token int32

// Bert represents a BERT model that can be used for encoding text into embeddings.
type Bert struct {
	handle  uintptr // Handle to the model
	threads int32   // Number of threads to use for encoding
	size    int32   // Size of the embeddings produced by the model
	maxtkn  int     // Maximum number of tokens
}

// New creates a new BERT model from the given model file.
func New(modelPath string) (*Bert, error) {
	ptr, err := exLoadModel(modelPath)
	if err != nil {
		return nil, err
	}

	size, err := exGetEmbeddingSize(ptr)
	if err != nil {
		return nil, err
	}

	maxtkn, err := exGetMaxTokens(ptr)
	if err != nil {
		return nil, err
	}

	return &Bert{
		handle:  ptr,
		threads: min(6, int32(runtime.NumCPU())),
		size:    int32(size),
		maxtkn:  maxtkn,
	}, nil
}

// MaxTokens returns the maximum number of tokens that can be processed by the model in a single input sequence.
func (m *Bert) MaxTokens() int {
	return m.maxtkn
}

// Size returns the size (dimensionality) of the embeddings produced by the model.
func (m *Bert) Size() int {
	return int(m.size)
}

// Close the BERT model and free all related resources.
func (m *Bert) Close() error {
	if m.handle != 0 {
		exFreeModel(m.handle)
		m.handle = 0
	}
	return nil
}

// EmbedText encodes a single piece of text into its embedding representation using the BERT model.
func (m *Bert) EmbedText(text string) ([]float32, error) {
	return exEmbedText(m.handle, text, m.threads, m.size)
}

// --------------------------------- Library Lookup ---------------------------------

// libBertPath is the path to the library.
var libBertPath string

func init() {
	var err error
	if libBertPath, err = findBertLibrary(); err != nil {
		panic(err)
	}

	if err := load(libBertPath); err != nil {
		panic(err)
	}
}

// findBertLibrary searches for the BERT dynamic library in standard system paths.
func findBertLibrary() (string, error) {
	switch runtime.GOOS {
	case "windows":
		return findLibrary("bert.dll", runtime.GOOS)
	case "darwin":
		return findLibrary("libbert.dylib", runtime.GOOS)
	default:
		return findLibrary("libbert.so", runtime.GOOS)
	}
}

// findLibrary searches for a dynamic library by name across standard system paths.
// It returns the full path to the library if found, or an error listing all searched paths.
func findLibrary(libName, goos string, dirs ...string) (string, error) {
	libExt, commonPaths := findLibDirs(goos)
	dirs = append(dirs, commonPaths...)

	// Append the correct extension if missing
	if !strings.HasSuffix(libName, libExt) {
		libName += libExt
	}

	// Include current working directory
	if cwd, err := os.Getwd(); err == nil {
		dirs = append(dirs, cwd)
	}

	// Iterate through directories and search for the library
	searched := make([]string, 0, len(dirs))
	for _, dir := range dirs {
		filename := filepath.Join(dir, libName)
		searched = append(searched, filename)
		if fi, err := os.Stat(filename); err == nil && !fi.IsDir() {
			return filename, nil // Library found
		}
	}

	// Construct error message listing all searched paths
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Library '%s' not found, checked following paths:\n", libName))
	for _, path := range searched {
		sb.WriteString(fmt.Sprintf(" - %s\n", path))
	}

	return "", errors.New(sb.String())
}

// findLibDirs returns the library extension, relevant environment path, and common library directories based on the OS.
func findLibDirs(goos string) (string, []string) {
	switch goos {
	case "windows":
		systemRoot := os.Getenv("SystemRoot")
		return ".dll", append(
			filepath.SplitList(os.Getenv("PATH")),
			filepath.Join(systemRoot, "System32"),
			filepath.Join(systemRoot, "SysWOW64"),
		)
	case "darwin":
		return ".dylib", append(
			filepath.SplitList(os.Getenv("DYLD_LIBRARY_PATH")),
			"/usr/lib",
			"/usr/local/lib",
		)
	default: // Unix/Linux
		return ".so", append(
			filepath.SplitList(os.Getenv("LD_LIBRARY_PATH")),
			"/lib",
			"/usr/lib",
			"/usr/local/lib",
		)
	}
}
