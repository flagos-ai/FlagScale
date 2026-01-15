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
