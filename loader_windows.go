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
	libBert     *windows.DLL
	fnLoad      *windows.Proc
	fnFree      *windows.Proc
	fnEmbedText *windows.Proc
	fnSize      *windows.Proc
	fnMaxToken  *windows.Proc
)

func load(libpath string) error {
	libBert = windows.MustLoadDLL(libpath)
	fnLoad = libBert.MustFindProc("bert_load_from_file")
	fnFree = libBert.MustFindProc("bert_free")
	fnEmbedText = libBert.MustFindProc("bert_encode")
	fnSize = libBert.MustFindProc("bert_n_embd")
	fnMaxToken = libBert.MustFindProc("bert_n_max_tokens")
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
