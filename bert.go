package bert

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unsafe"

	"github.com/ebitengine/purego"
)

// Bert represents a BERT model that can be used for encoding text into embeddings.
type Bert struct {
	handle  uintptr // Handle to the model
	threads int32   // Number of threads to use for encoding
	size    int32   // Size of the embeddings produced by the model
	maxtkn  int32   // Maximum number of tokens
}

// New creates a new BERT model from the given model file.
func New(modelPath string) (*Bert, error) {
	ptr := bert_load_from_file(modelPath)
	if ptr == 0 {
		return nil, fmt.Errorf("failed to load model from %s", modelPath)
	}

	return &Bert{
		handle:  ptr,
		threads: min(6, int32(runtime.NumCPU())),
		size:    bert_n_embd(ptr),
		maxtkn:  bert_n_max_tokens(ptr),
	}, nil
}

// MaxTokens returns the maximum number of tokens that can be processed by the model in a single input sequence.
func (m *Bert) MaxTokens() int {
	return int(m.maxtkn)
}

// Size returns the size (dimensionality) of the embeddings produced by the model.
func (m *Bert) Size() int {
	return int(m.size)
}

// Close the BERT model and free all related resources.
func (m *Bert) Close() error {
	if m.handle != 0 {
		bert_free(m.handle)
		m.handle = 0
	}
	return nil
}

// EmbedText encodes a single piece of text into its embedding representation using the BERT model.
func (m *Bert) EmbedText(text string) ([]float32, error) {
	embeddings := make([]float32, m.size)
	embedPtr := uintptr(unsafe.Pointer(&embeddings[0]))
	bert_encode(m.handle, m.threads, text, embedPtr)
	return embeddings, nil
}

// --------------------------------- Library Lookup ---------------------------------

// libBert is a handle to the dynamically loaded library.
var libBert uintptr

/*
typedef bert_ctx (*bert_load_from_file_t)(const char* fname);
typedef void (*bert_free_t)(bert_ctx ctx);
typedef void (*bert_encode_t)(bert_ctx ctx, int32_t n_threads, const char* text, float* embeddings);
typedef int32_t (*bert_n_embd_t)(bert_ctx ctx);
typedef int32_t (*bert_n_max_tokens_t)(bert_ctx ctx);
*/
var bert_load_from_file func(modelPath string) uintptr
var bert_free func(ptr uintptr)
var bert_encode func(ptr uintptr, threads int32, text string, embeddings uintptr)
var bert_n_embd func(ptr uintptr) int32
var bert_n_max_tokens func(ptr uintptr) int32

func init() {
	libBertPath, err := findBertLibrary()
	if err != nil {
		panic(err)
	}

	if libBert, err = load(libBertPath); err != nil {
		panic(err)
	}

	// Load the library functions
	purego.RegisterLibFunc(&bert_load_from_file, libBert, "bert_load_from_file")
	purego.RegisterLibFunc(&bert_free, libBert, "bert_free")
	purego.RegisterLibFunc(&bert_encode, libBert, "bert_encode")
	purego.RegisterLibFunc(&bert_n_embd, libBert, "bert_n_embd")
	purego.RegisterLibFunc(&bert_n_max_tokens, libBert, "bert_n_max_tokens")
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
