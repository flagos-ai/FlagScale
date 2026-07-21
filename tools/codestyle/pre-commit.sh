# Copyright 2026 FlagOS Contributors
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

set +x

if ! [[ $(pre-commit --version) == *"4.2.0"* ]]; then
    pip install pre-commit==4.2.0
fi

diff_files=$(git diff --name-only --diff-filter=ACMR ${BRANCH})
num_diff_files=$(echo "$diff_files" | wc -l)
echo -e "Different files between pr and ${BRANCH}:\n${diff_files}\n"

echo "Checking codestyle by pre-commit ..."
pre-commit run --files ${diff_files};check_error=$?

echo "*****************************************************************"
if [ ${check_error} != 0 ];then
    echo "Your PR codestyle check failed."
else
    echo "Your PR codestyle check passed."
fi
echo "*****************************************************************"

exit ${check_error}
