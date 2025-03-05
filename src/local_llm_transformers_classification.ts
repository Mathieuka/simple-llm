import {
	type PipelineType,
	type ProgressCallback,
	type TextClassificationPipeline,
	env,
	pipeline,
} from "@huggingface/transformers";

env.cacheDir = "./cache";

class MyClassificationPipeline {
	private task: PipelineType = "text-classification";
	private model = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";
	private instance: Promise<TextClassificationPipeline> | undefined = undefined;

	async getInstance() {
		if (this.instance === undefined) {
			this.instance = pipeline(this.task, this.model, {
				progress_callback: () => {
					// console.log("progress : ", progressInfo);
				},
				dtype: "fp32",
			}) as Promise<TextClassificationPipeline>;
		}

		return this.instance;
	}
}

const instance = await new MyClassificationPipeline().getInstance();

const response = await instance("Bad");
console.log(response);
