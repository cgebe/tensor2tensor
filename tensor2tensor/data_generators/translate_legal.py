# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
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

"""Data generator for legal translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

_TRAIN_DATASETS = {
    "cs-de": [
        [
            "https://transfer.sh/UCg6m/dcep.cs-de.tar.gz",
            ("dcep.cs-de.cs", "dcep.cs-de.de")
        ],
        [
            "https://transfer.sh/2sgQO/europarl-v7.cs-de.tar.gz",
            ("europarl-v7.cs-de.cs", "europarl-v7.cs-de.de")
        ],
        [
            "https://transfer.sh/8kO0W/jrc_acquis.cs-de.tar.gz",
            ("jrc_acquis.cs-de.cs", "jrc_acquis.cs-de.de")
        ],
    ],
    "cs-en": [
        [
            "https://transfer.sh/ob1Pj/dcep.cs-en.tar.gz",
            ("dcep.cs-en.cs", "dcep.cs-en.en")
        ],
        [
            "https://transfer.sh/2zBjZ/europarl-v7.cs-en.tar.gz",
            ("europarl-v7.cs-en.cs", "europarl-v7.cs-en.en")
        ],
        [
            "https://transfer.sh/mA2xT/jrc_acquis.cs-en.tar.gz",
            ("jrc_acquis.cs-en.cs", "jrc_acquis.cs-en.en")
        ]
    ],
    "cs-es": [
        [
            "https://transfer.sh/D0drq/dcep.cs-es.tar.gz",
            ("dcep.cs-es.cs", "dcep.cs-es.es")
        ],
        [
            "https://transfer.sh/15FAHV/europarl-v7.cs-es.tar.gz",
            ("europarl-v7.cs-es.cs", "europarl-v7.cs-es.es")
        ],
        [
            "https://transfer.sh/WDuqg/jrc_acquis.cs-es.tar.gz",
            ("jrc_acquis.cs-es.cs", "jrc_acquis.cs-es.es")
        ]
    ],
    "cs-fr": [
        [
            "https://transfer.sh/4qEUw/dcep.cs-fr.tar.gz",
            ("dcep.cs-fr.cs", "dcep.cs-fr.fr")
        ],
        [
            "https://transfer.sh/fESsl/europarl-v7.cs-fr.tar.gz",
            ("europarl-v7.cs-fr.cs", "europarl-v7.cs-fr.fr")
        ],
        [
            "https://transfer.sh/1Eb4X/jrc_acquis.cs-fr.tar.gz",
            ("jrc_acquis.cs-fr.cs", "jrc_acquis.cs-fr.fr")
        ]
    ],
    "cs-it": [
        [
            "https://transfer.sh/AYO1J/dcep.cs-it.tar.gz",
            ("dcep.cs-it.cs", "dcep.cs-it.it")
        ],
        [
            "https://transfer.sh/jK3GM/europarl-v7.cs-it.tar.gz",
            ("europarl-v7.cs-it.cs", "europarl-v7.cs-it.it")
        ],
        [
            "https://transfer.sh/10Ctre/jrc_acquis.cs-it.tar.gz",
            ("jrc_acquis.cs-it.cs", "jrc_acquis.cs-it.it")
        ]
    ],
    "cs-sv": [
        [
            "https://transfer.sh/TFK8b/dcep.cs-sv.tar.gz",
            ("dcep.cs-sv.cs", "dcep.cs-sv.sv")
        ],
        [
            "https://transfer.sh/jKFlc/europarl-v7.cs-sv.tar.gz",
            ("europarl-v7.cs-sv.cs", "europarl-v7.cs-sv.sv")
        ],
        [
            "https://transfer.sh/AUhx5/jrc_acquis.cs-sv.tar.gz",
            ("jrc_acquis.cs-sv.cs", "jrc_acquis.cs-sv.sv")
        ]
    ],
    "de-en": [
        [
            "https://transfer.sh/eO4Pt/dcep.de-en.tar.gz",
            ("dcep.de-en.de", "dcep.de-en.en")
        ],
        [
            "https://transfer.sh/REsAs/europarl-v7.de-en.tar.gz",
            ("europarl-v7.de-en.de", "europarl-v7.de-en.en")
        ],
        [
            "https://transfer.sh/Q5ofs/jrc_acquis.de-en.tar.gz",
            ("jrc_acquis.de-en.de", "jrc_acquis.de-en.en")
        ],
    ],
    "de-es": [
        [
            "https://transfer.sh/OsxIK/dcep.de-es.tar.gz",
            ("dcep.de-es.de", "dcep.de-es.es")
        ],
        [
            "https://transfer.sh/8tX9j/europarl-v7.de-es.tar.gz",
            ("europarl-v7.de-es.de", "europarl-v7.de-es.es")
        ],
        [
            "https://transfer.sh/dN4r5/jrc_acquis.de-es.tar.gz",
            ("jrc_acquis.de-es.de", "jrc_acquis.de-es.es")
        ]
    ],
    "de-fr": [
        [
            "https://transfer.sh/10MfiD/dcep.de-fr.tar.gz",
            ("dcep.de-fr.de", "dcep.de-fr.fr")
        ],
        [
            "https://transfer.sh/6VLp7/europarl-v7.de-fr.tar.gz",
            ("europarl-v7.de-fr.de", "europarl-v7.de-fr.fr")
        ],
        [
            "https://transfer.sh/CwbTq/jrc_acquis.de-fr.tar.gz",
            ("jrc_acquis.de-fr.de", "jrc_acquis.de-fr.fr")
        ]
    ],
    "de-it": [
        [
            "https://transfer.sh/mRiHi/dcep.de-it.tar.gz",
            ("dcep.de-it.de", "dcep.de-it.it")
        ],
        [
            "https://transfer.sh/15u0hX/europarl-v7.de-it.tar.gz",
            ("europarl-v7.de-it.de", "europarl-v7.de-it.it")
        ],
        [
            "https://transfer.sh/hsCyD/jrc_acquis.de-it.tar.gz",
            ("jrc_acquis.de-it.de", "jrc_acquis.de-it.it")
        ]
    ],
    "de-sv": [
        [
            "https://transfer.sh/iw1hd/dcep.de-sv.tar.gz",
            ("dcep.de-sv.de", "dcep.de-sv.sv")
        ],
        [
            "https://transfer.sh/w9nV0/europarl-v7.de-sv.tar.gz",
            ("europarl-v7.de-sv.de", "europarl-v7.de-sv.sv")
        ],
        [
            "https://transfer.sh/Asqe5/jrc_acquis.de-sv.tar.gz",
            ("jrc_acquis.de-sv.de", "jrc_acquis.de-sv.sv")
        ]
    ],
    "en-es": [
        [
            "https://transfer.sh/l0kdU/dcep.en-es.tar.gz",
            ("dcep.en-es.en", "dcep.en-es.es")
        ],
        [
            "https://transfer.sh/130oYb/europarl-v7.en-es.tar.gz",
            ("europarl-v7.en-es.en", "europarl-v7.en-es.es")
        ],
        [
            "https://transfer.sh/jwlyU/jrc_acquis.en-es.tar.gz",
            ("jrc_acquis.en-es.en", "jrc_acquis.en-es.es")
        ]
    ],
    "en-fr": [
        [
            "https://transfer.sh/xcXdz/dcep.en-fr.tar.gz",
            ("dcep.en-fr.en", "dcep.en-fr.fr")
        ],
        [
            "https://transfer.sh/QDzG9/europarl-v7.en-fr.tar.gz",
            ("europarl-v7.en-fr.en", "europarl-v7.en-fr.fr")
        ],
        [
            "https://transfer.sh/DsvZN/jrc_acquis.en-fr.tar.gz",
            ("jrc_acquis.en-fr.en", "jrc_acquis.en-fr.fr")
        ]
    ],
    "en-it": [
        [
            "https://transfer.sh/114FaH/dcep.en-it.tar.gz",
            ("dcep.en-it.en", "dcep.en-it.it")
        ],
        [
            "https://transfer.sh/11z82u/europarl-v7.en-it.tar.gz",
            ("europarl-v7.en-it.en", "europarl-v7.en-it.it")
        ],
        [
            "https://transfer.sh/W8RZS/jrc_acquis.en-it.tar.gz",
            ("jrc_acquis.en-it.en", "jrc_acquis.en-it.it")
        ]
    ],
    "en-sv": [
        [
            "https://transfer.sh/iGlhH/dcep.en-sv.tar.gz",
            ("dcep.en-sv.en", "dcep.en-sv.sv")
        ],
        [
            "https://transfer.sh/ncdjT/europarl-v7.en-sv.tar.gz",
            ("europarl-v7.en-sv.en", "europarl-v7.en-sv.sv")
        ],
        [
            "https://transfer.sh/DkE6w/jrc_acquis.en-sv.tar.gz",
            ("jrc_acquis.en-sv.en", "jrc_acquis.en-sv.sv")
        ]
    ],
    "es-fr": [
        [
            "https://transfer.sh/SK3gY/dcep.es-fr.tar.gz",
            ("dcep.es-fr.es", "dcep.es-fr.fr")
        ],
        [
            "https://transfer.sh/SZKt4/europarl-v7.es-fr.tar.gz",
            ("europarl-v7.es-fr.es", "europarl-v7.es-fr.fr")
        ],
        [
            "https://transfer.sh/eqwzv/jrc_acquis.es-fr.tar.gz",
            ("jrc_acquis.es-fr.es", "jrc_acquis.es-fr.fr")
        ]
    ],
    "es-it": [
        [
            "https://transfer.sh/10ravB/dcep.es-it.tar.gz",
            ("dcep.es-it.es", "dcep.es-it.it")
        ],
        [
            "https://transfer.sh/t8UfV/europarl-v7.es-it.tar.gz",
            ("europarl-v7.es-it.es", "europarl-v7.es-it.it")
        ],
        [
            "https://transfer.sh/Uj3En/jrc_acquis.es-it.tar.gz",
            ("jrc_acquis.es-it.es", "jrc_acquis.es-it.it")
        ]
    ],
    "es-sv": [
        [
            "https://transfer.sh/55fDW/dcep.es-sv.tar.gz",
            ("dcep.es-sv.es", "dcep.es-sv.sv")
        ],
        [
            "https://transfer.sh/LZVzV/europarl-v7.es-sv.tar.gz",
            ("europarl-v7.es-sv.es", "europarl-v7.es-sv.sv")
        ],
        [
            "https://transfer.sh/Y50xL/jrc_acquis.es-sv.tar.gz",
            ("jrc_acquis.es-sv.es", "jrc_acquis.es-sv.sv")
        ]
    ],
    "fr-it": [
        [
            "https://transfer.sh/MTOnW/dcep.fr-it.tar.gz",
            ("dcep.fr-it.fr", "dcep.fr-it.it")
        ],
        [
            "https://transfer.sh/R11Tv/europarl-v7.fr-it.tar.gz",
            ("europarl-v7.fr-it.fr", "europarl-v7.fr-it.it")
        ],
        [
            "https://transfer.sh/GcRaW/jrc_acquis.fr-it.tar.gz",
            ("jrc_acquis.fr-it.fr", "jrc_acquis.fr-it.it")
        ]
    ],
    "fr-sv": [
        [
            "https://transfer.sh/LJw2K/dcep.fr-sv.tar.gz",
            ("dcep.fr-sv.fr", "dcep.fr-sv.sv")
        ],
        [
            "https://transfer.sh/JEaD6/europarl-v7.fr-sv.tar.gz",
            ("europarl-v7.fr-sv.fr", "europarl-v7.fr-sv.sv")
        ],
        [
            "https://transfer.sh/obXyl/jrc_acquis.fr-sv.tar.gz",
            ("jrc_acquis.fr-sv.fr", "jrc_acquis.fr-sv.sv")
        ]
    ],
    "it-sv": [
        [
            "https://transfer.sh/WfNQS/dcep.it-sv.tar.gz",
            ("dcep.it-sv-it", "dcep.it-sv.sv")
        ],
        [
            "https://transfer.sh/h3CUa/europarl-v7.it-sv.tar.gz",
            ("europarl-v7.it-sv.it", "europarl-v7.it-sv.sv")
        ],
        [
            "https://transfer.sh/4oLE3/jrc_acquis.it-sv.tar.gz",
            ("jrc_acquis.it-sv.it", "jrc_acquis.it-sv.sv")
        ]
    ]
}

_TEST_DATASETS = {
    "cs-de": [
        [
            "https://transfer.sh/A1DDK/dcep.cs-de-test.tar.gz",
            ("dcep.cs-de-test.cs", "dcep.cs-de-test.de")
        ],
        [
            "https://transfer.sh/15OJTa/europarl-v7.cs-de-test.tar.gz",
            ("europarl-v7.cs-de-test.cs", "europarl-v7.cs-de-test.de")
        ],
        [
            "https://transfer.sh/ZaoIp/jrc_acquis.cs-de-test.tar.gz",
            ("jrc_acquis.cs-de-test.cs", "jrc_acquis.cs-de-test.de")
        ]
    ],
    "cs-en": [
        [
            "https://transfer.sh/u6K9T/dcep.cs-en-test.tar.gz",
            ("dcep.cs-en-test.cs", "dcep.cs-en-test.en")
        ],
        [
            "https://transfer.sh/INL7D/europarl-v7.cs-en-test.tar.gz",
            ("europarl-v7.cs-en-test.cs", "europarl-v7.cs-en-test.en")
        ],
        [
            "https://transfer.sh/H5ycC/jrc_acquis.cs-en-test.tar.gz",
            ("jrc_acquis.cs-en-test.cs", "jrc_acquis.cs-en-test.en")
        ]
    ],
    "cs-es": [
        [
            "https://transfer.sh/7SadY/dcep.cs-es-test.tar.gz",
            ("dcep.cs-es-test.cs", "dcep.cs-es-test.es")
        ],
        [
            "https://transfer.sh/C968u/europarl-v7.cs-es-test.tar.gz",
            ("europarl-v7.cs-es-test.cs", "europarl-v7.cs-es-test.es")
        ],
        [
            "https://transfer.sh/CthT9/jrc_acquis.cs-es-test.tar.gz",
            ("jrc_acquis.cs-es-test.cs", "jrc_acquis.cs-es-test.es")
        ]
    ],
    "cs-fr": [
        [
            "https://transfer.sh/V2Fw8/dcep.cs-fr-test.tar.gz",
            ("dcep.cs-fr-test.cs", "dcep.cs-fr-test.fr")
        ],
        [
            "https://transfer.sh/Extmq/europarl-v7.cs-fr-test.tar.gz",
            ("europarl-v7.cs-fr-test.cs", "europarl-v7.cs-fr-test.fr")
        ],
        [
            "https://transfer.sh/11h1mu/jrc_acquis.cs-fr-test.tar.gz",
            ("jrc_acquis.cs-fr-test.cs", "jrc_acquis.cs-fr-test.fr")
        ]
    ],
    "cs-it": [
        [
            "https://transfer.sh/xHCxP/dcep.cs-it-test.tar.gz",
            ("dcep.cs-it-test.cs", "dcep.cs-it-test.it")
        ],
        [
            "https://transfer.sh/dkQ4p/europarl-v7.cs-it-test.tar.gz",
            ("europarl-v7.cs-it-test.cs", "europarl-v7.cs-it-test.it")
        ],
        [
            "https://transfer.sh/ZnO8v/jrc_acquis.cs-it-test.tar.gz",
            ("jrc_acquis.cs-it-test.cs", "jrc_acquis.cs-it-test.it")
        ]
    ],
    "cs-sv": [
        [
            "https://transfer.sh/If8hi/dcep.cs-sv-test.tar.gz",
            ("dcep.cs-sv-test.cs", "dcep.cs-sv-test.sv")
        ],
        [
            "https://transfer.sh/yvCkW/europarl-v7.cs-sv-test.tar.gz",
            ("europarl-v7.cs-sv-test.cs", "europarl-v7.cs-sv-test.sv")
        ],
        [
            "https://transfer.sh/h1QGw/jrc_acquis.cs-sv-test.tar.gz",
            ("jrc_acquis.cs-sv-test.cs", "jrc_acquis.cs-sv-test.sv")
        ]
    ],
    "de-en": [
        [
            "https://transfer.sh/CIPrk/dcep.de-en-test.tar.gz",
            ("dcep.de-en-test.de", "dcep.de-en-test.en")
        ],
        [
            "https://transfer.sh/Mtzst/europarl-v7.de-en-test.tar.gz",
            ("europarl-v7.de-en-test.de", "europarl-v7.de-en-test.en")
        ],
        [
            "https://transfer.sh/g2XNL/jrc_acquis.de-en-test.tar.gz",
            ("jrc_acquis.de-en-test.de", "jrc_acquis.de-en-test.en")
        ]
    ],
    "de-es": [
        [
            "https://transfer.sh/PJ7H/dcep.de-es-test.tar.gz",
            ("dcep.de-es-test.de", "dcep.de-es-test.es")
        ],
        [
            "https://transfer.sh/loJK1/europarl-v7.de-es-test.tar.gz",
            ("europarl-v7.de-es-test.de", "europarl-v7.de-es-test.es")
        ],
        [
            "https://transfer.sh/14kv6J/jrc_acquis.de-es-test.tar.gz",
            ("jrc_acquis.de-es-test.de", "jrc_acquis.de-es-test.es")
        ]
    ],
    "de-fr": [
        [
            "https://transfer.sh/SKaTU/dcep.de-fr-test.tar.gz",
            ("dcep.de-fr-test.de", "dcep.de-fr-test.fr")
        ],
        [
            "https://transfer.sh/H86yL/europarl-v7.de-fr-test.tar.gz",
            ("europarl-v7.de-fr-test.de", "europarl-v7.de-fr-test.fr")
        ],
        [
            "https://transfer.sh/VkxyP/jrc_acquis.de-fr-test.tar.gz",
            ("jrc_acquis.de-fr-test.de", "jrc_acquis.de-fr-test.fr")
        ]
    ],
    "de-it": [
        [
            "https://transfer.sh/uB9rW/dcep.de-it-test.tar.gz",
            ("dcep.de-it-test.de", "dcep.de-it-test.it")
        ],
        [
            "https://transfer.sh/LpYYG/europarl-v7.de-it-test.tar.gz",
            ("europarl-v7.de-it-test.de", "europarl-v7.de-it-test.it")
        ],
        [
            "https://transfer.sh/106HDZ/jrc_acquis.de-it-test.tar.gz",
            ("jrc_acquis.de-it-test.de", "jrc_acquis.de-it-test.it")
        ]
    ],
    "de-sv": [
        [
            "https://transfer.sh/PoAam/dcep.de-sv-test.tar.gz",
            ("dcep.de-sv-test.de", "dcep.de-sv-test.sv")
        ],
        [
            "https://transfer.sh/SN3Oc/europarl-v7.de-sv-test.tar.gz",
            ("europarl-v7.de-sv-test.de", "europarl-v7.de-sv-test.sv")
        ],
        [
            "https://transfer.sh/CGpgh/jrc_acquis.de-sv-test.tar.gz",
            ("jrc_acquis.de-sv-test.de", "jrc_acquis.de-sv-test.sv")
        ]
    ],
    "en-es": [
        [
            "https://transfer.sh/m5OIV/dcep.en-es-test.tar.gz",
            ("dcep.en-es-test.en", "dcep.en-es-test.es")
        ],
        [
            "https://transfer.sh/PA3h9/europarl-v7.en-es-test.tar.gz",
            ("europarl-v7.en-es-test.en", "europarl-v7.en-es-test.es")
        ],
        [
            "https://transfer.sh/q4mCx/jrc_acquis.en-es-test.tar.gz",
            ("jrc_acquis.en-es-test.en", "jrc_acquis.en-es-test.es")
        ]
    ],
    "en-fr": [
        [
            "https://transfer.sh/Kibpt/dcep.en-fr-test.tar.gz",
            ("dcep.en-fr-test.en", "dcep.en-fr-test.fr")
        ],
        [
            "https://transfer.sh/127vl9/europarl-v7.en-fr-test.tar.gz",
            ("europarl-v7.en-fr-test.en", "europarl-v7.en-fr-test.fr")
        ],
        [
            "https://transfer.sh/12JQjr/jrc_acquis.en-fr-test.tar.gz",
            ("jrc_acquis.en-fr-test.en", "jrc_acquis.en-fr-test.fr")
        ]
    ],
    "en-it": [
        [
            "https://transfer.sh/y4Sax/dcep.en-it-test.tar.gz",
            ("dcep.en-it-test.en", "dcep.en-it-test.it")
        ],
        [
            "https://transfer.sh/2LUT2/europarl-v7.en-it-test.tar.gz",
            ("europarl-v7.en-it-test.en", "europarl-v7.en-it-test.it")
        ],
        [
            "https://transfer.sh/WeqCa/jrc_acquis.en-it-test.tar.gz",
            ("jrc_acquis.en-it-test.en", "jrc_acquis.en-it-test.it")
        ]
    ],
    "en-sv": [
        [
            "https://transfer.sh/1xy4j/dcep.en-sv-test.tar.gz",
            ("dcep.en-sv-test.en", "dcep.en-sv-test.sv")
        ],
        [
            "https://transfer.sh/6wwsf/europarl-v7.en-sv-test.tar.gz",
            ("europarl-v7.en-sv-test.en", "europarl-v7.en-sv-test.sv")
        ],
        [
            "https://transfer.sh/pIPoD/jrc_acquis.en-sv-test.tar.gz",
            ("jrc_acquis.en-sv-test.en", "jrc_acquis.en-sv-test.sv")
        ]
    ],
    "es-fr": [
        [
            "https://transfer.sh/OxEFi/dcep.es-fr-test.tar.gz",
            ("dcep.es-fr-test.es", "dcep.es-fr-test.fr")
        ],
        [
            "https://transfer.sh/rtY29/europarl-v7.es-fr-test.tar.gz",
            ("europarl-v7.es-fr-test.es", "europarl-v7.es-fr-test.fr")
        ],
        [
            "https://transfer.sh/sDmzu/jrc_acquis.es-fr-test.tar.gz",
            ("jrc_acquis.es-fr-test.es", "jrc_acquis.es-fr-test.fr")
        ]
    ],
    "es-it": [
        [
            "https://transfer.sh/ZCDPp/dcep.es-it-test.tar.gz",
            ("dcep.es-it-test.es", "dcep.es-it-test.it")
        ],
        [
            "https://transfer.sh/oORFH/europarl-v7.es-it-test.tar.gz",
            ("europarl-v7.es-it-test.es", "europarl-v7.es-it-test.it")
        ],
        [
            "https://transfer.sh/DARu0/jrc_acquis.es-it-test.tar.gz",
            ("jrc_acquis.es-it-test.es", "jrc_acquis.es-it-test.it")
        ]
    ],
    "es-sv": [
        [
            "https://transfer.sh/5mO8g/dcep.es-sv-test.tar.gz",
            ("dcep.es-sv-test.es", "dcep.es-sv-test.sv")
        ],
        [
            "https://transfer.sh/dhWfn/europarl-v7.es-sv-test.tar.gz",
            ("europarl-v7.es-sv-test.es", "europarl-v7.es-sv-test.sv")
        ],
        [
            "https://transfer.sh/1jxI5/jrc_acquis.es-sv-test.tar.gz",
            ("jrc_acquis.es-sv-test.es", "jrc_acquis.es-sv-test.sv")
        ]
    ],
    "fr-it": [
        [
            "https://transfer.sh/jrXB2/dcep.fr-it-test.tar.gz",
            ("dcep.fr-it-test.fr", "dcep.fr-it-test.it")
        ],
        [
            "https://transfer.sh/9F4tL/europarl-v7.fr-it-test.tar.gz",
            ("europarl-v7.fr-it-test.fr", "europarl-v7.fr-it-test.it")
        ],
        [
            "https://transfer.sh/rf4Ne/jrc_acquis.fr-it-test.tar.gz",
            ("jrc_acquis.fr-it-test.fr", "jrc_acquis.fr-it-test.it")
        ]
    ],
    "fr-sv": [
        [
            "https://transfer.sh/MWfMp/dcep.fr-sv-test.tar.gz",
            ("dcep.fr-sv-test.fr", "dcep.fr-sv-test.sv")
        ],
        [
            "https://transfer.sh/yfwNf/europarl-v7.fr-sv-test.tar.gz",
            ("europarl-v7.fr-sv-test.fr", "europarl-v7.fr-sv-test.sv")
        ],
        [
            "https://transfer.sh/TaSJS/jrc_acquis.fr-sv-test.tar.gz",
            ("jrc_acquis.fr-sv-test.fr", "jrc_acquis.fr-sv-test.sv")
        ]
    ],
    "it-sv": [
        [
            "https://transfer.sh/SqKx1/dcep.it-sv-test.tar.gz",
            ("dcep.it-sv-test.it", "dcep.it-sv-test.sv")
        ],
        [
            "https://transfer.sh/10xMyv/europarl-v7.it-sv-test.tar.gz",
            ("europarl-v7.it-sv-test.it", "europarl-v7.it-sv-test.sv")
        ],
        [
            "https://transfer.sh/E06za/jrc_acquis.it-sv-test.tar.gz",
            ("jrc_acquis.it-sv-test.it", "jrc_acquis.it-sv-test.sv")
        ]
    ]
}


@registry.register_problem
class TranslateCsdeLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.csde"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["cs-de"])
        datasets = _TRAIN_DATASETS["cs-de"] if train else _TEST_DATASETS["cs-de"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_csde_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.CS_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.DE_TOK


@registry.register_problem
class TranslateCsenLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.csen"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["cs-en"])
        datasets = _TRAIN_DATASETS["cs-en"] if train else _TEST_DATASETS["cs-en"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_csen_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.CS_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_TOK


@registry.register_problem
class TranslateCsesLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.cses"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["cs-es"])
        datasets = _TRAIN_DATASETS["cs-es"] if train else _TEST_DATASETS["cs-es"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_cses_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.CS_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.ES_TOK


@registry.register_problem
class TranslateCsfrLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.csfr"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["cs-fr"])
        datasets = _TRAIN_DATASETS["cs-fr"] if train else _TEST_DATASETS["cs-fr"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_csfr_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.CS_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.FR_TOK


@registry.register_problem
class TranslateCsitLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.csit"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["cs-it"])
        datasets = _TRAIN_DATASETS["cs-it"] if train else _TEST_DATASETS["cs-it"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_csit_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.CS_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.IT_TOK


@registry.register_problem
class TranslateCssvLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.cssv"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["cs-sv"])
        datasets = _TRAIN_DATASETS["cs-sv"] if train else _TEST_DATASETS["cs-sv"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_cssv_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.CS_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.SV_TOK


@registry.register_problem
class TranslateDeenLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.deen"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["de-en"])
        datasets = _TRAIN_DATASETS["de-en"] if train else _TEST_DATASETS["de-en"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_deen_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.DE_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.EN_TOK


@registry.register_problem
class TranslateDeesLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.dees"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["de-es"])
        datasets = _TRAIN_DATASETS["de-es"] if train else _TEST_DATASETS["de-es"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_dees_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.DE_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.ES_TOK


@registry.register_problem
class TranslateDefrLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.defr"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["de-fr"])
        datasets = _TRAIN_DATASETS["de-fr"] if train else _TEST_DATASETS["de-fr"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_defr_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.DE_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.FR_TOK


@registry.register_problem
class TranslateDeitLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.deit"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["de-it"])
        datasets = _TRAIN_DATASETS["de-it"] if train else _TEST_DATASETS["de-it"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_deit_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.DE_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.IT_TOK


@registry.register_problem
class TranslateDesvLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.desv"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["de-sv"])
        datasets = _TRAIN_DATASETS["de-sv"] if train else _TEST_DATASETS["de-sv"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_desv_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.DE_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.SV_TOK


@registry.register_problem
class TranslateEnesLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.enes"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["en-es"])
        datasets = _TRAIN_DATASETS["en-es"] if train else _TEST_DATASETS["en-es"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_enes_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.ES_TOK


@registry.register_problem
class TranslateEnfrLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.enfr"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["en-fr"])
        datasets = _TRAIN_DATASETS["en-fr"] if train else _TEST_DATASETS["en-fr"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_enfr_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.FR_TOK


@registry.register_problem
class TranslateEnitLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.enit"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["en-it"])
        datasets = _TRAIN_DATASETS["en-it"] if train else _TEST_DATASETS["en-it"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_enit_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.IT_TOK


@registry.register_problem
class TranslateEnsvLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.ensv"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["en-sv"])
        datasets = _TRAIN_DATASETS["en-sv"] if train else _TEST_DATASETS["en-sv"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_ensv_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.SV_TOK


@registry.register_problem
class TranslateEsfrLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.esfr"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["es-fr"])
        datasets = _TRAIN_DATASETS["es-fr"] if train else _TEST_DATASETS["es-fr"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_esfr_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.ES_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.FR_TOK


@registry.register_problem
class TranslateEsitLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.enit"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["es-it"])
        datasets = _TRAIN_DATASETS["es-it"] if train else _TEST_DATASETS["es-it"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_enit_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.ES_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.IT_TOK


@registry.register_problem
class TranslateEssvLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.essv"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["es-sv"])
        datasets = _TRAIN_DATASETS["es-sv"] if train else _TEST_DATASETS["es-sv"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_essv_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.ES_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.SV_TOK


@registry.register_problem
class TranslateFritLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.frit"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["fr-it"])
        datasets = _TRAIN_DATASETS["fr-it"] if train else _TEST_DATASETS["fr-it"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_frit_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.FR_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.IT_TOK


@registry.register_problem
class TranslateFrsvLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.frsv"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["fr-sv"])
        datasets = _TRAIN_DATASETS["fr-sv"] if train else _TEST_DATASETS["fr-sv"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_frsv_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.FR_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.SV_TOK


@registry.register_problem
class TranslateItsvLegal8k(translate.TranslateProblem):
    """Problem spec for Legal translation."""

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192

    @property
    def vocab_name(self):
        return "vocab.itsv"

    def generator(self, data_dir, tmp_dir, train):
        symbolizer_vocab = generator_utils.get_or_generate_vocab(
            data_dir, tmp_dir, self.vocab_file, self.targeted_vocab_size, _TRAIN_DATASETS["it-sv"])
        datasets = _TRAIN_DATASETS["it-sv"] if train else _TEST_DATASETS["it-sv"]
        tag = "train" if train else "dev"
        data_path = translate.compile_data(
            tmp_dir, datasets, "legal_itsv_tok_%s" % tag)
        return translate.token_generator(data_path + ".lang1", data_path + ".lang2", symbolizer_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.IT_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.SV_TOK
