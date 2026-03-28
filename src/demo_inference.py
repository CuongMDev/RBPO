from mepo_inference import MePOModel
model = MePOModel()

query = "Explain RAG in simple terms"
result = model.inference(query)

print(result)
