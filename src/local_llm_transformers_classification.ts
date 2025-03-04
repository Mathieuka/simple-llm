import {
	type PipelineType,
	type ProgressCallback,
	type TextClassificationPipeline,
	pipeline,
} from "@huggingface/transformers";

class MyClassificationPipeline {
	private task: PipelineType = "text-classification";
	private model = "Xenova/distilbert-base-uncased-finetuned-sst-2-english";
	private instance: Promise<TextClassificationPipeline> | undefined = undefined;

	async getInstance(
		progress_callback: ProgressCallback | undefined = undefined,
	) {
		if (this.instance === undefined) {
			this.instance = pipeline(this.task, this.model, {
				progress_callback,
				dtype: "fp32",
			}) as Promise<TextClassificationPipeline>;
		}

		return this.instance;
	}
}

const instance = await new MyClassificationPipeline().getInstance();

const response = await instance("Try it!");
console.log(response);
