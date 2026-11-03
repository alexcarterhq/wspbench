"""Provider-client settings are separate from Inspect generation settings."""


def provider_client_options(model_config: dict) -> dict:
    # Inspect's GenerateConfig disables its outer loop, not OpenAI SDK retries.
    # One reservation must cover exactly one potentially billable HTTP request.
    if model_config.get("provider")!="openai":return {}
    configured=model_config.get("model_args",{})
    if not isinstance(configured,dict) or set(configured)-{"responses_api","service_tier","streaming"}:
        raise ValueError("OpenAI model_args permits only responses_api, explicit streaming and the fixed default service tier")
    # A Boolean avoids Inspect's automatic streaming-rejection fallback call.
    streaming=configured.get("streaming",False)
    if type(streaming) is not bool:raise ValueError("streaming must be explicitly Boolean")
    options={"max_retries":0,"streaming":streaming}
    if "responses_api" in configured:
        if type(configured["responses_api"]) is not bool:
            raise ValueError("responses_api must be Boolean")
        options["responses_api"]=configured["responses_api"]
    if "service_tier" in configured:
        if configured["service_tier"]!="default":raise ValueError("Only the default standard service tier is supported")
        options["service_tier"]="default"
    return options


def provider_generate_options(model_config: dict) -> dict:
    options={"max_retries":0,"cache":False,"cache_prompt":False}
    if model_config.get("provider")=="openai":
        # None asks Inspect to probe organization support with an extra,
        # unmetered Responses generation. Suppress summaries explicitly.
        options["reasoning_summary"]="none"
    return options
