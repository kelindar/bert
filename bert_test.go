package bert

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

// BenchmarkBert/encode-24         	     314	   3664379 ns/op	    1648 B/op	       3 allocs/op
func BenchmarkBert(b *testing.B) {
	ctx := loadModel()
	defer ctx.Close()

	text := "This is a test sentence we are going to generate embeddings for."

	b.Run("encode", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, err := ctx.EmbedText(text)
			assert.NoError(b, err)
		}
	})
}

func loadModel() *Bert {
	mod, _ := filepath.Abs("dist/minilm12-q4.bin")
	ctx, err := New(mod)
	if err != nil {
		panic(err)
	}
	return ctx
}

func TestFindLibrary_Windows(t *testing.T) {
	path, err := findLibrary("bert.dll", "windows", "dist/win-x64")
	assert.NotEmpty(t, path)
	assert.NoError(t, err)
}

func TestFindLibrary_Linux(t *testing.T) {
	path, err := findLibrary("libbert.so", "linux", "dist/linux-x64")
	assert.NotEmpty(t, path)
	assert.NoError(t, err)
}

func cosine(a, b []float32) float32 {
	var dot, na, nb float32
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	return dot / (na * nb)
}
