import { HfInference } from "@huggingface/inference";

const hf = process.env.HF;
console.log("%c LOG hf", "background: #222; color: #bada55", hf);
const client = new HfInference(hf);

const chatCompletion = await client.chatCompletion({
	model: "HuggingFaceH4/zephyr-7b-beta",
	messages: [
		{
			role: "user",
			content: "What is the capital of France?",
		},
	],
	provider: "hf-inference",
	max_tokens: 10,
});

console.log(chatCompletion.choices[0].message);
