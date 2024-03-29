# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module exposes pretty printing functionality."""

from dataclasses import dataclass

import jax
import rich
import rich.traceback as traceback
from rich.console import Console


#####
# Pretty printing
#####


@dataclass
class ADEVJAXConsole:
    rich_console: Console

    def print(self, obj):
        self.rich_console.print(obj)

    def inspect(self, obj, **kwargs):
        rich.inspect(obj, console=self.rich_console, **kwargs)

    def help(self, obj):
        rich.inspect(
            obj,
            console=self.rich_console,
            methods=True,
            help=True,
            value=False,
            private=False,
            dunder=False,
        )


def pretty(show_locals=True, max_frames=30, suppress=[jax], **kwargs):
    rich.pretty.install()
    traceback.install(
        show_locals=show_locals,
        max_frames=max_frames,
        suppress=suppress,
    )

    return ADEVJAXConsole(Console(soft_wrap=True, **kwargs))
