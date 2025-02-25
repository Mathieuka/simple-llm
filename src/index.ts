import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";

const model = new ChatOpenAI({ model: "gpt-3.5-turbo" });

const messages = [
	new SystemMessage("Translate the following from English into French"),
	new HumanMessage("hi!"),
];

const res = await model.invoke(messages);
