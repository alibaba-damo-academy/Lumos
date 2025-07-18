# Copyright 2024 Meta Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from transformers.utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available

_import_structure = {
    "configuration_chameleon": ["ChameleonConfig", "ChameleonVQVAEConfig"],
    "processing_chameleon": ["ChameleonProcessor"],
}


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_chameleon"] = [
        "ChameleonForConditionalGeneration",
        "ChameleonModel",
        "ChameleonPreTrainedModel",
        "ChameleonVQVAE",
    ]
    _import_structure["modeling_chameleon_mm_rope"] = [
        "ChameleonMMRoPEForConditionalGeneration",
    ]

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_chameleon"] = ["ChameleonImageProcessor"]


if TYPE_CHECKING:
    from .configuration_chameleon import ChameleonConfig, ChameleonVQVAEConfig
    from .processing_chameleon import ChameleonProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_chameleon import (
            ChameleonForConditionalGeneration,
            ChameleonModel,
            ChameleonPreTrainedModel,
            ChameleonVQVAE,
        )
        from .modeling_chameleon_mm_rope import (
            ChameleonMMRoPEForConditionalGeneration,
        )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_chameleon import ChameleonImageProcessor


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
