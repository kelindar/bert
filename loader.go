//go:build !windows
// +build !windows

package bert

/*
#cgo linux LDFLAGS: -ldl
#cgo darwin LDFLAGS: -ldl

#include <dlfcn.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

typedef void* bert_ctx;

typedef bert_ctx (*bert_load_from_file_t)(const char* fname);
typedef void (*bert_free_t)(bert_ctx ctx);
typedef void (*bert_encode_t)(bert_ctx ctx, int32_t n_threads, const char* text, float* embeddings);
typedef void (*bert_encode_batch_t)(bert_ctx ctx, int32_t n_threads, int32_t n_batch_size, int32_t n_inputs, const char** texts, float** embeddings);
typedef void (*bert_tokenize_t)(bert_ctx ctx, const char* text, int32_t* tokens, int32_t* n_tokens, int32_t n_max_tokens);
typedef void (*bert_eval_t)(bert_ctx ctx, int32_t n_threads, int32_t* tokens, int32_t n_tokens, float* embeddings);
typedef void (*bert_eval_batch_t)(bert_ctx ctx, int32_t n_threads, int32_t n_batch_size, int32_t** batch_tokens, int32_t* n_tokens, float** batch_embeddings);
typedef int32_t (*bert_n_embd_t)(bert_ctx ctx);
typedef int32_t (*bert_n_max_tokens_t)(bert_ctx ctx);
typedef const char* (*bert_vocab_id_to_token_t)(bert_ctx ctx, int32_t id);

static void* bert_lib_handle = NULL;

static bert_load_from_file_t bert_load_from_file_func = NULL;
static bert_free_t bert_free_func = NULL;
static bert_encode_t bert_encode_func = NULL;
static bert_encode_batch_t bert_encode_batch_func = NULL;
static bert_tokenize_t bert_tokenize_func = NULL;
static bert_eval_t bert_eval_func = NULL;
static bert_eval_batch_t bert_eval_batch_func = NULL;
static bert_n_embd_t bert_n_embd_func = NULL;
static bert_n_max_tokens_t bert_n_max_tokens_func = NULL;
static bert_vocab_id_to_token_t bert_vocab_id_to_token_func = NULL;

static char bert_error_msg[256];

const char* bert_get_error() {
    return bert_error_msg;
}

int bert_load_library(const char* lib_path) {
    bert_lib_handle = dlopen(lib_path, RTLD_LAZY | RTLD_LOCAL);
    if (!bert_lib_handle) {
        snprintf(bert_error_msg, sizeof(bert_error_msg), "Failed to load library: %s", dlerror());
        return -1;
    }

    bert_load_from_file_func = (bert_load_from_file_t)dlsym(bert_lib_handle, "bert_load_from_file");
    bert_free_func = (bert_free_t)dlsym(bert_lib_handle, "bert_free");
    bert_encode_func = (bert_encode_t)dlsym(bert_lib_handle, "bert_encode");
    bert_encode_batch_func = (bert_encode_batch_t)dlsym(bert_lib_handle, "bert_encode_batch");
    bert_tokenize_func = (bert_tokenize_t)dlsym(bert_lib_handle, "bert_tokenize");
    bert_eval_func = (bert_eval_t)dlsym(bert_lib_handle, "bert_eval");
    bert_eval_batch_func = (bert_eval_batch_t)dlsym(bert_lib_handle, "bert_eval_batch");
    bert_n_embd_func = (bert_n_embd_t)dlsym(bert_lib_handle, "bert_n_embd");
    bert_n_max_tokens_func = (bert_n_max_tokens_t)dlsym(bert_lib_handle, "bert_n_max_tokens");
    bert_vocab_id_to_token_func = (bert_vocab_id_to_token_t)dlsym(bert_lib_handle, "bert_vocab_id_to_token");

    if (!bert_load_from_file_func || !bert_free_func || !bert_encode_func || !bert_encode_batch_func ||
        !bert_tokenize_func || !bert_eval_func || !bert_eval_batch_func || !bert_n_embd_func ||
        !bert_n_max_tokens_func || !bert_vocab_id_to_token_func) {
        snprintf(bert_error_msg, sizeof(bert_error_msg), "Failed to load symbols: %s", dlerror());
        return -1;
    }

    return 0;
}

bert_ctx bert_load_model(const char* model_path) {
    return bert_load_from_file_func(model_path);
}

void bert_free_model(bert_ctx ctx) {
    bert_free_func(ctx);
}

int32_t bert_n_embd(bert_ctx ctx) {
    return bert_n_embd_func(ctx);
}

int32_t bert_n_max_tokens(bert_ctx ctx) {
    return bert_n_max_tokens_func(ctx);
}

void bert_encode(bert_ctx ctx, int32_t n_threads, const char* text, float* embeddings) {
    bert_encode_func(ctx, n_threads, text, embeddings);
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// load loads the shared library at the given path.
func load(libPath string) error {
	cPath := C.CString(libPath)
	defer C.free(unsafe.Pointer(cPath))

	if ret := C.bert_load_library(cPath); ret != 0 {
		return fmt.Errorf("failed to load %s: %s", libPath, C.GoString(C.bert_get_error()))
	}
	return nil
}

// exLoadModel loads the BERT model from the specified path.
func exLoadModel(modelPath string) (uintptr, error) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	ctx := C.bert_load_model(cPath)
	if ctx == nil {
		return 0, fmt.Errorf("bert_load_model failed")
	}
	return uintptr(ctx), nil
}

// exGetEmbeddingSize retrieves the embedding size of the model.
func exGetEmbeddingSize(handle uintptr) (int, error) {
	return int(C.bert_n_embd(C.bert_ctx(handle))), nil
}

// exGetMaxTokens retrieves the maximum number of tokens the model can handle.
func exGetMaxTokens(handle uintptr) (int, error) {
	return int(C.bert_n_max_tokens(C.bert_ctx(handle))), nil
}

// exFreeModel frees the loaded BERT model.
func exFreeModel(handle uintptr) {
	C.bert_free_model(C.bert_ctx(handle))
}

// exEmbedText encodes a single piece of text into embeddings.
func exEmbedText(handle uintptr, text string, nThreads, size int32) ([]float32, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	embeddings := make([]float32, size)
	C.bert_encode(
		C.bert_ctx(handle),
		C.int32_t(nThreads),
		cText,
		(*C.float)(unsafe.Pointer(&embeddings[0])),
	)
	return embeddings, nil
}
