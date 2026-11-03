import asyncio
import pytest

from wspbench.provider_options import provider_client_options,provider_generate_options


def test_openai_sdk_retries_are_disabled_separately_from_inspect(tmp_path,monkeypatch):
    from inspect_ai.model import get_model,GenerateConfig
    monkeypatch.setenv("INSPECT_CACHE_DIR",str(tmp_path/"cache"))
    model=get_model("openai/gpt-6-astra",api_key="fixture-no-network",memoize=False,
                    config=GenerateConfig(**provider_generate_options({"provider":"openai"})),**provider_client_options({"provider":"openai"}))
    assert model.config.max_retries==0
    assert model.config.reasoning_summary=="none"
    assert model.api.client.max_retries==0
    assert model.api.streaming is False
    asyncio.run(model.api.client.close())


def test_unknown_provider_does_not_receive_openai_specific_sdk_arguments():
    assert provider_client_options({"provider":"anthropic"})=={}


def test_explicit_responses_transport_is_honored_without_reenabling_retries(tmp_path,monkeypatch):
    from inspect_ai.model import get_model,GenerateConfig
    monkeypatch.setenv("INSPECT_CACHE_DIR",str(tmp_path/"cache"))
    config={"provider":"openai","model_args":{"responses_api":True,"service_tier":"default","streaming":True}}
    model=get_model("openai/gpt-4.1",api_key="fixture-no-network",memoize=False,
                    config=GenerateConfig(**provider_generate_options(config)),**provider_client_options(config))
    assert model.api.responses_api is True
    assert model.api.service_tier=="default"
    assert model.api.streaming is True
    assert model.api.client.max_retries==0
    asyncio.run(model.api.client.close())


@pytest.mark.parametrize("arguments",[{"max_retries":2},{"api_key":"untrusted"},{"responses_api":"true"},{"service_tier":"priority"},{"streaming":"auto"},{"streaming":1},[]])
def test_openai_client_configuration_cannot_smuggle_sdk_overrides(arguments):
    with pytest.raises(ValueError):provider_client_options({"provider":"openai","model_args":arguments})


@pytest.mark.parametrize("disable_probe",[False,True])
def test_real_inspect_adapter_makes_exactly_one_generation_when_probe_disabled(tmp_path,monkeypatch,disable_probe):
    from openai.types.responses import Response
    from inspect_ai.model import get_model,GenerateConfig,ChatMessageUser
    monkeypatch.setenv("INSPECT_CACHE_DIR",str(tmp_path/"cache"))
    captured=[]
    async def create(**request):
        captured.append(request)
        return Response.model_validate({"id":"resp_fixture","created_at":0,"model":"gpt-6-astra",
            "object":"response","output":[{"id":"msg_fixture","type":"message","role":"assistant",
            "status":"completed","content":[{"type":"output_text","text":"OK","annotations":[],"logprobs":[]}]}],
            "parallel_tool_calls":False,"tool_choice":"none","tools":[],"status":"completed",
            "usage":{"input_tokens":7,"output_tokens":3,"total_tokens":10,
                     "input_tokens_details":{"cached_tokens":0,"cache_write_tokens":0},
                     "output_tokens_details":{"reasoning_tokens":0}}})
    model_config={"provider":"openai","model_args":{"responses_api":True,"service_tier":"default"}}
    options=provider_generate_options(model_config)
    if not disable_probe:options.pop("reasoning_summary")
    model=get_model("openai/gpt-6-astra",api_key="fixture-no-network",memoize=False,
                    config=GenerateConfig(**options),**provider_client_options(model_config))
    monkeypatch.setattr(model.api.client.responses,"create",create)
    async def run():
        try:
            output=await model.api.generate([ChatMessageUser(content="Return OK")],[],"none",GenerateConfig(max_tokens=12,**options))
            return output[0] if isinstance(output,tuple) else output
        finally:await model.api.client.close()
    output=asyncio.run(run())
    assert output.completion=="OK"
    assert len(captured)==(1 if disable_probe else 2)
    assert captured[-1]["max_output_tokens"]==12
    assert captured[-1]["service_tier"]=="default"
    if disable_probe:
        assert all(not isinstance(request["input"],str) for request in captured)
    else:
        # This reproduces the SDK capability call that was outside benchmark
        # accounting: a separate scalar-string input without an output cap.
        assert isinstance(captured[0]["input"],str)
        assert "max_output_tokens" not in captured[0]
