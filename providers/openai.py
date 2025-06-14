"""OpenAI model provider implementation."""

from typing import Optional

from .base import (
    FixedTemperatureConstraint,
    ModelCapabilities,
    ModelResponse,
    ProviderType,
    RangeTemperatureConstraint,
)
from .openai_compatible import OpenAICompatibleProvider


class OpenAIModelProvider(OpenAICompatibleProvider):
    """Official OpenAI API provider (api.openai.com)."""

    # Model configurations
    SUPPORTED_MODELS = {
        "o3": {
            "context_window": 200_000,  # 200K tokens
            "supports_extended_thinking": False,
        },
        "o4-mini": {
            "context_window": 200_000,  # 200K tokens
            "supports_extended_thinking": False,
        },
        "o3-pro": {
            "context_window": 200_000,  # 200K tokens
            "supports_extended_thinking": False,
        },
    }

    # Valid additional parameters for OpenAI API
    VALID_API_PARAMS = ["top_p", "frequency_penalty", "presence_penalty", "seed", "stop"]
    
    # O-series models that don't support sampling parameters
    O_SERIES_MODELS = {"o3", "o4-mini", "o3-pro"}

    def __init__(self, api_key: str, **kwargs):
        """Initialize OpenAI provider with API key."""
        # Set default OpenAI base URL, allow override for regions/custom endpoints
        kwargs.setdefault("base_url", "https://api.openai.com/v1")
        super().__init__(api_key, **kwargs)

    def get_capabilities(self, model_name: str) -> ModelCapabilities:
        """Get capabilities for a specific OpenAI model."""
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported OpenAI model: {model_name}")

        config = self.SUPPORTED_MODELS[model_name]

        # Define temperature constraints per model
        if model_name in ["o3", "o4-mini", "o3-pro"]:
            # O3 models only support temperature=1.0
            temp_constraint = FixedTemperatureConstraint(1.0)
        else:
            # Other OpenAI models support 0.0-2.0 range
            temp_constraint = RangeTemperatureConstraint(0.0, 2.0, 0.7)

        return ModelCapabilities(
            provider=ProviderType.OPENAI,
            model_name=model_name,
            friendly_name="OpenAI",
            context_window=config["context_window"],
            supports_extended_thinking=config["supports_extended_thinking"],
            supports_system_prompts=True,
            supports_streaming=True,
            supports_function_calling=True,
            temperature_constraint=temp_constraint,
        )

    def generate_content(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using OpenAI model."""
        # Validate parameters
        self.validate_parameters(model_name, temperature)

        # Use different API endpoint for o3-pro model
        if model_name == "o3-pro":
            return self._generate_with_responses_api(
                prompt=prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                **kwargs,
            )

        # Filter out unsupported parameters for O-series models
        if model_name in self.O_SERIES_MODELS:
            # Remove sampling parameters that O-series models don't support
            filtered_kwargs = {
                k: v for k, v in kwargs.items() 
                if k not in ["top_p", "frequency_penalty", "presence_penalty"]
            }
            # O-series models only support temperature=1.0, so we ignore the temperature parameter
            # The parent class will handle this through validate_parameters
            return super().generate_content(
                prompt=prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                temperature=1.0,  # O-series models only support temperature=1.0
                max_output_tokens=max_output_tokens,
                **filtered_kwargs,
            )
        
        # Use the parent class implementation for other models
        return super().generate_content(
            prompt=prompt,
            model_name=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

    def _generate_with_responses_api(
        self,
        prompt: str,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> ModelResponse:
        """Generate content using OpenAI v1/responses API (for o3-pro model)."""
        import logging

        import requests

        # Prepare the full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Add organization header if configured
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        # Build the request payload
        # Note: v1/responses API uses "input" instead of "prompt"
        payload = {
            "model": model_name,
            "input": full_prompt,
        }
        
        # O-series models don't support sampling parameters
        # (temperature, top_p, presence_penalty, frequency_penalty)
        if model_name not in self.O_SERIES_MODELS:
            payload["temperature"] = temperature

        # Add max output tokens if specified
        # Note: v1/responses uses "max_output_tokens" not "max_tokens"
        if max_output_tokens:
            payload["max_output_tokens"] = max_output_tokens

        # Add any additional parameters
        for key, value in kwargs.items():
            # Skip empty keys and only include known OpenAI parameters
            if key and key in self.VALID_API_PARAMS:
                # Skip sampling parameters for O-series models
                if model_name in self.O_SERIES_MODELS and key in ["top_p", "frequency_penalty", "presence_penalty"]:
                    logging.debug(f"Skipping unsupported parameter '{key}' for O-series model {model_name}")
                    continue
                payload[key] = value
            elif not key:
                # Log empty key detection
                logging.warning(f"Detected empty key in kwargs with value: {value}")

        # Determine the API endpoint
        base_url = self.base_url or "https://api.openai.com"
        # Remove any trailing slashes
        base_url = base_url.rstrip("/")

        # For OpenAI API, we expect the base URL to not include /v1
        # If it ends with /v1, remove it to avoid duplication
        if base_url.endswith("/v1"):
            base_url = base_url[:-3].rstrip("/")

        endpoint = f"{base_url}/v1/responses"

        try:
            # Make API request
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()

            # Extract content from response
            # The v1/responses API has a different structure than chat completions
            content = ""
            if "output" in data and isinstance(data["output"], list):
                # Look for the message output item
                for item in data["output"]:
                    if item.get("type") == "message" and "content" in item:
                        # Extract text from content array
                        for content_item in item["content"]:
                            if content_item.get("type") == "output_text" and "text" in content_item:
                                content = content_item["text"]
                                break
                        if content:
                            break

            # Fallback for unexpected response structure
            if not content:
                logging.warning(
                    f"Unexpected response structure from v1/responses API for {model_name}. "
                    f"Response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
                )
                content = str(data)

            # Extract usage information if available
            usage = {}
            if "usage" in data:
                # v1/responses uses different field names
                usage = {
                    "input_tokens": data["usage"].get("input_tokens", 0),
                    "output_tokens": data["usage"].get("output_tokens", 0),
                    "total_tokens": data["usage"].get("total_tokens", 0),
                }

            return ModelResponse(
                content=content,
                usage=usage,
                model_name=model_name,
                friendly_name="OpenAI",
                provider=ProviderType.OPENAI,
                metadata={
                    "model": data.get("model", model_name),
                    "id": data.get("id", ""),
                    "created": data.get("created_at", 0),
                    "api_endpoint": "v1/responses",
                    "status": data.get("status", ""),
                },
            )

        except Exception as e:
            # Handle different types of exceptions
            error_msg = f"OpenAI API error for model {model_name} (v1/responses): {str(e)}"

            if isinstance(e, requests.exceptions.RequestException):
                if hasattr(e, "response") and e.response is not None:
                    try:
                        error_data = e.response.json()
                        error_msg = f"OpenAI API error for model {model_name} (v1/responses): {error_data}"
                    except Exception:
                        error_msg = f"OpenAI API error for model {model_name} (v1/responses): {e.response.text}"

            logging.error(error_msg)
            raise RuntimeError(error_msg) from e
    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return ProviderType.OPENAI

    def validate_model_name(self, model_name: str) -> bool:
        """Validate if the model name is supported."""
        return model_name in self.SUPPORTED_MODELS

    def supports_thinking_mode(self, model_name: str) -> bool:
        """Check if the model supports extended thinking mode."""
        # Currently no OpenAI models support extended thinking
        # This may change with future O3 models
        return False
