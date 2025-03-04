import {
	type PipelineType,
	type ProgressCallback,
	type TextGenerationOutput,
	type TextGenerationPipeline,
	type TextGenerationSingle,
	env,
	pipeline,
} from "@huggingface/transformers";

env.cacheDir = "./cache";

class MyGenerationPipeline {
	private task: PipelineType = "text-generation";
	private model = "HuggingFaceTB/SmolLM2-360M-Instruct";
	private instance: Promise<TextGenerationPipeline> | undefined = undefined;

	async getInstance(
		progress_callback: ProgressCallback | undefined = undefined,
	) {
		if (this.instance === undefined) {
			this.instance = pipeline(this.task, this.model, {
				progress_callback,
				dtype: "fp32",
			}) as Promise<TextGenerationPipeline>;
		}

		return this.instance;
	}

	async generateResponse(prompt: string) {
		const instance = await this.getInstance();
		const response = (await instance(prompt, {
			max_new_tokens: 50,
			do_sample: true,
			temperature: 0.7,
		})) as TextGenerationSingle[];

		return response[0].generated_text;
	}
}

// Utilisation
const myPipeline = new MyGenerationPipeline();

async function main() {
	const prompt = "Human: What is the capital of France ?\nAssistant:";
	const response = await myPipeline.generateResponse(prompt);
	console.log(response);
}

await main();
