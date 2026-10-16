// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0.

function parseCommand(body) {
  const match = /^\/ci run ([a-z][a-z0-9_-]*)$/.exec(body.trim());
  return match ? match[1] : null;
}

module.exports = async ({ github, context, core }) => {
  const target = parseCommand(context.payload.comment.body);
  if (!target) return;
  const { data: permission } = await github.rest.repos.getCollaboratorPermissionLevel({
    ...context.repo, username: context.payload.comment.user.login
  });
  if (!['admin', 'maintain', 'write'].includes(permission.permission)) {
    core.notice('CI requests require repository write permission.');
    return;
  }
  const { data: pr } = await github.rest.pulls.get({
    ...context.repo, pull_number: context.payload.issue.number
  });
  if (pr.state !== 'open' || !pr.head.repo) {
    core.setFailed('The PR must be open and its head repository available.');
    return;
  }
  // Workflow and platform list both come from the trusted default branch.
  const { data: configs } = await github.rest.repos.getContent({
    ...context.repo, path: '.github/configs', ref: context.payload.repository.default_branch
  });
  const platforms = configs.filter(file => file.name.endsWith('.yml') &&
    !['template.yml', 'image_sources.yml', 'platforms.yml'].includes(file.name))
    .map(file => file.name.slice(0, -4));
  if (target !== 'all' && !platforms.includes(target)) {
    core.setFailed(`Unknown platform: ${target}. Available: all, ${platforms.join(', ')}`);
    return;
  }
  await github.rest.actions.createWorkflowDispatch({
    ...context.repo, workflow_id: 'ci-request.yml',
    ref: context.payload.repository.default_branch,
    inputs: { pr_number: String(pr.number), head_sha: pr.head.sha, platform: target }
  });
  await github.rest.issues.createComment({
    ...context.repo, issue_number: pr.number,
    body: `CI request accepted: **${target}**, commit \`${pr.head.sha}\`. ` +
      `[View requested runs](${context.payload.repository.html_url}/actions/workflows/ci-request.yml). ` +
      'This supplemental run does not replace the CI Complete merge gate.'
  });
};
module.exports.parseCommand = parseCommand;
