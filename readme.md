# Advanced multimodal RAG with Microsoft Azure AI Search and LlamaIndex

This example demonstrates the implementation of a sophisticated multimodal Retrieval-Augmented Generation (RAG) system using Azure AI Search, Azure OpenAI services, and LlamaIndex. The system processes both text and images from PDF documents, storing them in Azure Blob Storage and Azure AI Search for efficient retrieval.

This example uses LlamaParse to parse PDF documents. LlamaParse is a Generative AI native document parser that can parse complex document data for any downstream Large Language Model (LLM) use case, such as RAGs.

## Microsoft Azure account and LlamaCloud (LlamaParse) API requirements

For this example, you need a Microsoft Azure account. You can [try Azure for free](https://azure.microsoft.com/en-us/pricing/purchase-options/azure-account).

You also need a LlamaCloud API key to use LlamaParse. You can [get a free LlamaCloud API key](https://cloud.llamaindex.ai/login) for up to 1000 pages per day.

## Preconditions

In Microsoft Azure, you first need to create these resources:

1. Create a "Resource Group".
2. Create an "Azure OpenAI" endpoint. From there, you can start the "Azure AI Foundry".
3. In "Azure AI Foundry", create a new project with your OpenAI endpoints.
4. In "Azure AI Foundry", deploy a model like "gpt-4o-mini" for chat completion.
5. In "Azure AI Foundry", deploy a model like "text-embedding-3-small" for embedding.

In the associated Azure Blob Storage account, you need to set the following permissions:

1. Click on "Settings" and then click on "Configuration". Set "Allow Blob anonymous access" to "Enabled", and click on "Save".
2. For the Azure Blob container: Click on "Change access level", select "Blob (anonymous read access for blobs only)", and click on "OK".

Note: In a production environment with sensitive data, you should use SAS tokens instead of enabling anonymous access.

## Usage

You need to fill in the required pieces of information in the `.env.example` file and then rename this file to just `.env` (remove the `.example` ending).

Running the `main.py` script ingests the document `AI_Report.pdf` and gets your RAG system ready to use. You can use your RAG system from command line, or you can use the much nicer Chainlit web interface.

## Running the Chainlit interface

To interact with this RAG system through a user-friendly chat interface, you can use Chainlit. The Chainlit interface provides a conversational way to query your documents and see responses with both text and images.

### Steps to run the Chainlit app:

1. Open your terminal in the project directory.
2. Run the following command:
   ```
   chainlit run interface.py -w
   ```
   The `-w` flag enables hot-reloading for development.

The Chainlit interface will be available at `http://localhost:8000` by default. You can interact with your RAG system through a modern, responsive web interface that makes it easy to visualize both text and image responses.

You can enter your prompt to get a text answer as well as an image of the relevant page of the source document, for example:

```
How can AI improve ESG reporting in the investment management industry?
```

![alt text](https://github.com/user-attachments/assets/025ac03d-595c-4dad-8470-9be304274cc2 "Multimodal RAG")
