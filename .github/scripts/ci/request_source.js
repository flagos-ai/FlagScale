// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0.
module.exports = async ({ github, context, core, prNumber, headSha }) => {
  if (!/^[1-9][0-9]*$/.test(prNumber) || !/^[a-f0-9]{40}$/.test(headSha)) {
    throw new Error('Expected a PR number and a full commit SHA');
  }
  const { data: pr } = await github.rest.pulls.get({
    ...context.repo, pull_number: Number(prNumber)
  });
  if (pr.state !== 'open' || !pr.head.repo || pr.head.sha !== headSha) {
    throw new Error('PR is closed, unavailable, or its head changed; request CI again');
  }
  core.setOutput('repository', pr.head.repo.full_name);
  core.setOutput('sha', pr.head.sha);
  await core.summary.addHeading('Requested CI source')
    .addRaw(`PR #${pr.number}: ${pr.head.repo.full_name}@${pr.head.sha}\n`).write();
};
