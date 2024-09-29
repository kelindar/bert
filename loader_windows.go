//go:build windows
// +build windows

package bert

import (
	_ "embed"
	"fmt"
	"sort"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

var (
	libBert            *windows.DLL
	fnLoad             *windows.Proc
	fnFree             *windows.Proc
	fnEmbedText        *windows.Proc
	fnEmbedTextBatch   *windows.Proc
	fnTokenize         *windows.Proc
	fnEmbedTokens      *windows.Proc
	fnEmbedTokensBatch *windows.Proc
	fnSize             *windows.Proc
	fnMaxToken         *windows.Proc
	fnTokenString      *windows.Proc
)

func load(libpath string) error {
	libBert = windows.MustLoadDLL(libpath)
	fnLoad = libBert.MustFindProc("bert_load_from_file")
	fnFree = libBert.MustFindProc("bert_free")
	fnEmbedText = libBert.MustFindProc("bert_encode")
	fnEmbedTextBatch = libBert.MustFindProc("bert_encode_batch")
	fnTokenize = libBert.MustFindProc("bert_tokenize")
	fnEmbedTokens = libBert.MustFindProc("bert_eval")
	fnEmbedTokensBatch = libBert.MustFindProc("bert_eval_batch")
	fnSize = libBert.MustFindProc("bert_n_embd")
	fnMaxToken = libBert.MustFindProc("bert_n_max_tokens")
	fnTokenString = libBert.MustFindProc("bert_vocab_id_to_token")
	return nil
}

func exLoadModel(modelPath string) (uintptr, error) {
	modelPathPtr, err := syscall.BytePtrFromString(modelPath)
	if err != nil {
		return 0, fmt.Errorf("failed to convert model path to C string: %v", err)
	}

	ptr, _, callErr := fnLoad.Call(uintptr(unsafe.Pointer(modelPathPtr)))
	if ptr == 0 {
		return 0, fmt.Errorf("bert_load_from_file failed: %v", callErr)
	}
	return ptr, nil
}

func exGetEmbeddingSize(handle uintptr) (int, error) {
	size, _, err := fnSize.Call(handle)
	if err != windows.ERROR_SUCCESS && err != nil {
		return 0, err
	}
	return int(size), nil
}

func exGetMaxTokens(handle uintptr) (int, error) {
	maxtkn, _, err := fnMaxToken.Call(handle)
	if err != windows.ERROR_SUCCESS && err != nil {
		return 0, err
	}
	return int(maxtkn), nil
}

func exFreeModel(handle uintptr) {
	fnFree.Call(handle)
}

func exEmbedText(handle uintptr, text string, nThreads, size int32) ([]float32, error) {
	textPtr, err := syscall.BytePtrFromString(text)
	if err != nil {
		return nil, fmt.Errorf("failed to convert text to C string: %v", err)
	}

	embeddings := make([]float32, size)
	_, _, callErr := fnEmbedText.Call(
		handle,
		uintptr(nThreads),
		uintptr(unsafe.Pointer(textPtr)),
		uintptr(unsafe.Pointer(&embeddings[0])),
	)
	if callErr != syscall.Errno(0) {
		return nil, fmt.Errorf("bert_encode failed: %v", callErr)
	}
	return embeddings, nil
}

func exEmbedTextBatch(handle uintptr, texts []string, nThreads, batchSize, size int32) ([][]float32, error) {
	nInputs := int32(len(texts))
	textPtrs := make([]*byte, nInputs)
	for i, text := range texts {
		textPtr, err := syscall.BytePtrFromString(text)
		if err != nil {
			return nil, fmt.Errorf("failed to convert text to C string: %v", err)
		}
		textPtrs[i] = textPtr
	}

	embeddingsList := make([][]float32, nInputs)
	embeddingsPtrs := make([]uintptr, nInputs)
	for i := range embeddingsList {
		embeddings := make([]float32, size)
		embeddingsList[i] = embeddings
		embeddingsPtrs[i] = uintptr(unsafe.Pointer(&embeddings[0]))
	}

	_, _, callErr := fnEmbedTextBatch.Call(
		handle,
		uintptr(nThreads),
		uintptr(batchSize),
		uintptr(nInputs),
		uintptr(unsafe.Pointer(&textPtrs[0])),
		uintptr(unsafe.Pointer(&embeddingsPtrs[0])),
	)
	if callErr != syscall.Errno(0) {
		return nil, fmt.Errorf("bert_encode_batch failed: %v", callErr)
	}
	return embeddingsList, nil
}

func exTokenizeText(handle uintptr, text string, maxTokens int) ([]Token, error) {
	textPtr, err := syscall.BytePtrFromString(text)
	if err != nil {
		return nil, fmt.Errorf("failed to convert text to C string: %v", err)
	}

	tokens := make([]Token, maxTokens)
	nTokens := int32(0)

	_, _, callErr := fnTokenize.Call(
		handle,
		uintptr(unsafe.Pointer(textPtr)),
		uintptr(unsafe.Pointer(&tokens[0])),
		uintptr(unsafe.Pointer(&nTokens)),
		uintptr(int32(maxTokens)),
	)
	if callErr != syscall.Errno(0) {
		return nil, fmt.Errorf("bert_tokenize failed: %v", callErr)
	}
	return tokens[:nTokens], nil
}

func exEmbedTokens(handle uintptr, tokens []Token, nThreads, size int32) ([]float32, error) {
	nTokens := int32(len(tokens))
	embeddings := make([]float32, size)

	_, _, err := fnEmbedTokens.Call(
		handle,
		uintptr(nThreads),
		uintptr(unsafe.Pointer(&tokens[0])),
		uintptr(nTokens),
		uintptr(unsafe.Pointer(&embeddings[0])),
	)
	if err != windows.ERROR_SUCCESS && err != nil {
		return nil, err
	}
	return embeddings, nil
}

func exEmbedTokensBatch(handle uintptr, tokenBatches [][]Token, nThreads, batchSize, size int32) ([][]float32, error) {
	nInputs := int32(len(tokenBatches))
	if nInputs == 0 {
		return nil, fmt.Errorf("no input token sequences provided")
	}

	// Ensure the longest input is first
	sort.Slice(tokenBatches, func(i, j int) bool {
		return len(tokenBatches[i]) > len(tokenBatches[j])
	})

	tokenPtrs := make([]uintptr, nInputs)
	nTokens := make([]int32, nInputs)

	for i, tokens := range tokenBatches {
		nTokens[i] = int32(len(tokens))
		tokenPtrs[i] = uintptr(unsafe.Pointer(&tokens[0]))
	}

	embeddingsList := make([][]float32, nInputs)
	embeddingsPtrs := make([]uintptr, nInputs)

	for i := range embeddingsList {
		embeddings := make([]float32, size)
		embeddingsList[i] = embeddings
		embeddingsPtrs[i] = uintptr(unsafe.Pointer(&embeddings[0]))
	}

	tokenPtrsPtr := uintptr(unsafe.Pointer(&tokenPtrs[0]))
	nTokensPtr := uintptr(unsafe.Pointer(&nTokens[0]))
	embeddingsPtrsPtr := uintptr(unsafe.Pointer(&embeddingsPtrs[0]))

	_, _, callErr := fnEmbedTokensBatch.Call(
		handle,
		uintptr(nThreads),
		uintptr(batchSize),
		tokenPtrsPtr,
		nTokensPtr,
		embeddingsPtrsPtr,
	)
	if callErr != syscall.Errno(0) {
		return nil, fmt.Errorf("bert_eval_batch failed: %v", callErr)
	}

	return embeddingsList, nil
}

func exTokenToString(handle uintptr, id Token) (string, error) {
	ret, _, callErr := fnTokenString.Call(handle, uintptr(id))
	if ret == 0 {
		return "", fmt.Errorf("bert_vocab_id_to_token failed: %v", callErr)
	}
	return asString(ret), nil
}

// Helper function to convert C char* to Go string.
func asString(cstr uintptr) string {
	var bytes []byte
	ptr := cstr
	for {
		b := *(*byte)(unsafe.Pointer(ptr))
		if b == 0 {
			break
		}
		bytes = append(bytes, b)
		ptr++
	}
	return string(bytes)
}
