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

void bert_encode_batch(bert_ctx ctx, int32_t n_threads, int32_t n_batch_size, int32_t n_inputs, const char** texts, float** embeddings) {
    bert_encode_batch_func(ctx, n_threads, n_batch_size, n_inputs, texts, embeddings);
}

void bert_tokenize(bert_ctx ctx, const char* text, int32_t* tokens, int32_t* n_tokens, int32_t n_max_tokens) {
    bert_tokenize_func(ctx, text, tokens, n_tokens, n_max_tokens);
}

void bert_eval(bert_ctx ctx, int32_t n_threads, int32_t* tokens, int32_t n_tokens, float* embeddings) {
    bert_eval_func(ctx, n_threads, tokens, n_tokens, embeddings);
}

void bert_eval_batch(bert_ctx ctx, int32_t n_threads, int32_t n_batch_size, int32_t** batch_tokens, int32_t* n_tokens, float** batch_embeddings) {
    bert_eval_batch_func(ctx, n_threads, n_batch_size, batch_tokens, n_tokens, batch_embeddings);
}

const char* bert_vocab_id_to_token(bert_ctx ctx, int32_t id) {
    return bert_vocab_id_to_token_func(ctx, id);
}

// C helper function to set n_tokens at a specific index
void set_n_token_at(int32_t* n_tokens_array, int32_t index, int32_t value) {
    n_tokens_array[index] = value;
}
*/
import "C"

import (
	"fmt"
	"sort"
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

// exEmbedTextBatch encodes multiple texts into embeddings in a batch.
func exEmbedTextBatch(handle uintptr, texts []string, nThreads, batchSize, size int32) ([][]float32, error) {
	nInputs := int32(len(texts))
	if nInputs == 0 {
		return nil, fmt.Errorf("no input texts provided")
	}

	// Allocate C array for texts
	cTexts := C.malloc(C.size_t(nInputs) * C.size_t(unsafe.Sizeof(uintptr(0))))
	if cTexts == nil {
		return nil, fmt.Errorf("failed to allocate memory for texts")
	}
	defer C.free(cTexts)

	cTextsArr := (*[1 << 30]*C.char)(cTexts)
	cStrs := make([]*C.char, nInputs)
	for i, txt := range texts {
		cStr := C.CString(txt)
		cStrs[i], cTextsArr[i] = cStr, cStr
	}
	defer func() {
		for _, cStr := range cStrs {
			C.free(unsafe.Pointer(cStr))
		}
	}()

	// Allocate C array for embeddings pointers
	cEmbs := C.malloc(C.size_t(nInputs) * C.size_t(unsafe.Sizeof(uintptr(0))))
	if cEmbs == nil {
		return nil, fmt.Errorf("failed to allocate memory for embeddings pointers")
	}
	defer C.free(cEmbs)

	cEmbsArr := (*[1 << 30]*C.float)(cEmbs)
	embList := make([][]float32, nInputs)
	cEmbPtrs := make([]*C.float, nInputs)

	for i := int32(0); i < nInputs; i++ {
		embList[i] = make([]float32, size)
		cEmb := C.malloc(C.size_t(size) * C.size_t(unsafe.Sizeof(C.float(0))))
		if cEmb == nil {
			return nil, fmt.Errorf("failed to allocate memory for embeddings[%d]", i)
		}
		cEmbPtrs[i], cEmbsArr[i] = (*C.float)(cEmb), (*C.float)(cEmb)
		defer C.free(unsafe.Pointer(cEmb))
	}

	// Call the C function
	C.bert_encode_batch(
		C.bert_ctx(handle),
		C.int32_t(nThreads),
		C.int32_t(batchSize),
		C.int32_t(nInputs),
		(**C.char)(cTexts),
		(**C.float)(cEmbs),
	)

	// Copy the embeddings back to Go slices
	for i := int32(0); i < nInputs; i++ {
		cEmb := cEmbPtrs[i]
		goEmb := embList[i]
		for j := int32(0); j < size; j++ {
			offset := uintptr(j) * unsafe.Sizeof(C.float(0))
			goEmb[j] = float32(*(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(cEmb)) + offset)))
		}
	}

	return embList, nil
}

// exTokenizeText tokenizes the given text into tokens.
func exTokenizeText(handle uintptr, text string, maxTokens int) ([]Token, error) {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	tokens := make([]C.int32_t, maxTokens)
	var nTokens C.int32_t

	C.bert_tokenize(
		C.bert_ctx(handle),
		cText,
		&tokens[0],
		&nTokens,
		C.int32_t(maxTokens),
	)

	result := make([]Token, nTokens)
	for i := 0; i < int(nTokens); i++ {
		result[i] = Token(tokens[i])
	}
	return result, nil
}

// exEmbedTokens embeds a sequence of tokens into embeddings.
func exEmbedTokens(handle uintptr, tokens []Token, nThreads, size int32) ([]float32, error) {
	nTokens := int32(len(tokens))
	embeddings := make([]float32, size)

	cTokens := make([]C.int32_t, nTokens)
	for i, tok := range tokens {
		cTokens[i] = C.int32_t(tok)
	}

	C.bert_eval(
		C.bert_ctx(handle),
		C.int32_t(nThreads),
		&cTokens[0],
		C.int32_t(nTokens),
		(*C.float)(unsafe.Pointer(&embeddings[0])),
	)

	return embeddings, nil
}

// exEmbedTokensBatch embeds multiple token sequences into embeddings in a batch.
func exEmbedTokensBatch(handle uintptr, tokenBatches [][]Token, nThreads, batchSize, size int32) ([][]float32, error) {
	nInputs := int32(len(tokenBatches))
	if nInputs == 0 {
		return nil, fmt.Errorf("no input token sequences provided")
	}

	// Sort token batches by length in descending order
	sort.Slice(tokenBatches, func(i, j int) bool {
		return len(tokenBatches[i]) > len(tokenBatches[j])
	})

	// Allocate C array for batch_tokens
	cBatchTokens := C.malloc(C.size_t(nInputs) * C.size_t(unsafe.Sizeof(uintptr(0))))
	if cBatchTokens == nil {
		return nil, fmt.Errorf("failed to allocate memory for batch_tokens")
	}
	defer C.free(cBatchTokens)

	cBatchTokensArr := (*[1 << 30]*C.int32_t)(cBatchTokens)
	nTokensArr := make([]C.int32_t, nInputs)

	for i, batch := range tokenBatches {
		nTokensArr[i] = C.int32_t(len(batch))
		if len(batch) > 0 {
			cTokens := C.malloc(C.size_t(len(batch)) * C.size_t(unsafe.Sizeof(C.int32_t(0))))
			if cTokens == nil {
				return nil, fmt.Errorf("failed to allocate memory for tokens[%d]", i)
			}
			cTokensArr := (*[1 << 30]C.int32_t)(cTokens)
			for j, tok := range batch {
				cTokensArr[j] = C.int32_t(tok)
			}
			cBatchTokensArr[i] = (*C.int32_t)(cTokens)
			defer C.free(unsafe.Pointer(cTokens))
		} else {
			cBatchTokensArr[i] = nil
		}
	}

	// Allocate C array for n_tokens
	cNTokens := C.malloc(C.size_t(nInputs) * C.size_t(unsafe.Sizeof(C.int32_t(0))))
	if cNTokens == nil {
		return nil, fmt.Errorf("failed to allocate memory for n_tokens")
	}
	defer C.free(cNTokens)

	cNTokensArr := (*C.int32_t)(cNTokens)
	for i := int32(0); i < nInputs; i++ {
		C.set_n_token_at(cNTokensArr, C.int32_t(i), nTokensArr[i])
	}

	// Allocate C array for batch_embeddings pointers
	cBatchEmbs := C.malloc(C.size_t(nInputs) * C.size_t(unsafe.Sizeof(uintptr(0))))
	if cBatchEmbs == nil {
		return nil, fmt.Errorf("failed to allocate memory for batch_embeddings pointers")
	}
	defer C.free(cBatchEmbs)

	cBatchEmbsArr := (*[1 << 30]*C.float)(cBatchEmbs)
	embList := make([][]float32, nInputs)
	cEmbPtrs := make([]*C.float, nInputs)

	for i := int32(0); i < nInputs; i++ {
		embList[i] = make([]float32, size)
		cEmb := C.malloc(C.size_t(size) * C.size_t(unsafe.Sizeof(C.float(0))))
		if cEmb == nil {
			return nil, fmt.Errorf("failed to allocate memory for embeddings[%d]", i)
		}
		cEmbPtrs[i], cBatchEmbsArr[i] = (*C.float)(cEmb), (*C.float)(cEmb)
		defer C.free(unsafe.Pointer(cEmb))
	}

	// Call the C function
	C.bert_eval_batch(
		C.bert_ctx(handle),
		C.int32_t(nThreads),
		C.int32_t(batchSize),
		(**C.int32_t)(cBatchTokens),
		(*C.int32_t)(cNTokens),
		(**C.float)(cBatchEmbs),
	)

	// Copy the embeddings back to Go slices
	for i := int32(0); i < nInputs; i++ {
		cEmb := cEmbPtrs[i]
		goEmb := embList[i]
		for j := int32(0); j < size; j++ {
			offset := uintptr(j) * unsafe.Sizeof(C.float(0))
			goEmb[j] = float32(*(*C.float)(unsafe.Pointer(uintptr(unsafe.Pointer(cEmb)) + offset)))
		}
	}

	return embList, nil
}

// exTokenToString converts a token ID to its corresponding string.
func exTokenToString(handle uintptr, id Token) (string, error) {
	cStr := C.bert_vocab_id_to_token(C.bert_ctx(handle), C.int32_t(id))
	if cStr == nil {
		return "", fmt.Errorf("bert_vocab_id_to_token failed")
	}
	return C.GoString(cStr), nil
}
