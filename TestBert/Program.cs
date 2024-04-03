using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;

var options = SessionOptions.MakeSessionOptionWithCudaProvider();

// This test is from https://github.com/microsoft/onnxruntime/issues/18854
// The model was downloaded from https://www.dropbox.com/scl/fi/vyxnknulralskuzevtcqn/model.fp16.onnx?rlkey=ftk96zuzqwwk4xb8f883w5w2y&dl=1
var session = new InferenceSession("D:\\csharp\\TestBert\\model.fp16.onnx", options);

// Create sample inputs (vocabulary size is 128,000)
var inputIds = Enumerable.Range(0, 27 * 52).Select(_ => Random.Shared.NextInt64(3, 128000)).ToArray();
var mask = Enumerable.Repeat(1L, 27 * 52).ToArray();
var tokenTypeIds = Enumerable.Repeat(0L, 27 * 52).ToArray();

// Try running multiple inputs in parallel
int test_count = 20;
int thread_count = 10;
for (int i = 0; i < test_count; i++)
{
    Enumerable.Range(0, thread_count).AsParallel().ForAll(_ =>
    {
        var output = session.Run(new[] {
        NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(inputIds, new[] { 27, 52 })),
        NamedOnnxValue.CreateFromTensor("token_type_ids", new DenseTensor<long>(mask, new[] { 27, 52 })),
        NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(tokenTypeIds, new[] { 27, 52 })),
    });
    });
}

Console.WriteLine("Finished testing {0} threads {1} times. No problem found.", thread_count, test_count);
return;
