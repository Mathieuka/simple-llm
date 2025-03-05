import {
	AutoModel,
	AutoModelForSequenceClassification,
	AutoTokenizer,
	type EncodingSingle,
	type PreTrainedTokenizer,
	env,
	softmax,
} from "@huggingface/transformers";

env.cacheDir = "./cache";

const checkpoint = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";

const tokenizer: PreTrainedTokenizer =
	await AutoTokenizer.from_pretrained(checkpoint);

const rawInput = "i'm sad";

const inputs: EncodingSingle = tokenizer(rawInput, {
	padding: true,
	truncation: true,
	return_tensors: "pt",
});

const model = await AutoModelForSequenceClassification.from_pretrained(
	checkpoint,
	{
		dtype: "fp32",
	},
);
console.log(
	"%c LOG model",
	"background: #222; color: #bada55",
	model.config.label2id,
);

const res = await model(inputs);

const logitsData = res.logits.ort_tensor.cpuData;

console.log(
	"%c LOG logitsData",
	"background: #222; color: #bada55",
	logitsData,
);

console.log(
	"%c LOG softmax",
	"background: #222; color: #bada55",
	softmax(logitsData),
);
