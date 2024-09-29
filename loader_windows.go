//go:build windows
// +build windows

package bert

import (
	_ "embed"
	"fmt"
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
