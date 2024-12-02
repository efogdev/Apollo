from .language_model.configs import ApolloQwenConfig, ApolloLlamaConfig, ApolloGemmaConfig
AVAILABLE_MODELS = {
    "apollo_llm": "ApolloForCausalLM, ApolloConfig",
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except ImportError:
        import traceback

        traceback.print_exc()
        print(f"Failed to import {model_name} from apollo.language_model.{model_name}")
        pass


