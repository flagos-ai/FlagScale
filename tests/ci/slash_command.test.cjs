// Copyright 2026 FlagOS Contributors
// Licensed under the Apache License, Version 2.0.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const command = require('../../.github/scripts/ci/slash_command.js');

function fixture(permission = 'write', body = '/ci run ascend') {
  const calls = [];
  const sha = 'a'.repeat(40);
  const context = { repo: { owner: 'team', repo: 'FlagScale' }, payload: {
    comment: { body, user: { login: 'member' } }, issue: { number: 42 },
    repository: { default_branch: 'main', html_url: 'https://github.com/team/FlagScale' }
  } };
  const github = { rest: {
    repos: {
      getCollaboratorPermissionLevel: async () => ({ data: { permission } }),
      getContent: async args => {
        calls.push(['configs', args]);
        return { data: ['ascend.yml', 'cuda.yml', 'ppu.yml', 'template.yml', 'image_sources.yml']
          .map(name => ({ name })) };
      }
    },
    pulls: { get: async () => ({ data: { number: 42, state: 'open', head: {
      sha, repo: { full_name: 'contributor/FlagScale' }
    } } }) },
    actions: { createWorkflowDispatch: async args => calls.push(['dispatch', args]) },
    issues: { createComment: async args => calls.push(['comment', args]) }
  } };
  const core = { notice: value => calls.push(['notice', value]), setFailed: value => calls.push(['failed', value]) };
  return { github, context, core, calls, sha };
}

for (const input of ['/ci run ascend; echo injected', 'please /ci run all', '/ci run ../../secret',
  '/ci run all\n/ci run cuda', '/ci run', '/ci run ASCEND']) {
  test(`reject malformed command ${JSON.stringify(input)}`, () => {
    assert.equal(command.parseCommand(input), null);
  });
}

test('dispatch pins current PR commit and trusted workflow branch', async () => {
  const f = fixture();
  await command(f);
  const dispatched = f.calls.find(([name]) => name === 'dispatch')[1];
  assert.equal(dispatched.workflow_id, 'ci-request.yml');
  assert.equal(dispatched.ref, 'main');
  assert.deepEqual(dispatched.inputs, { pr_number: '42', head_sha: f.sha, platform: 'ascend' });
  assert.equal(f.calls.find(([name]) => name === 'configs')[1].ref, 'main');
  assert.match(f.calls.find(([name]) => name === 'comment')[1].body, /does not replace/);
});

for (const role of ['read', 'triage', 'none']) {
  test(`no dispatch for ${role} permission`, async () => {
    const f = fixture(role);
    await command(f);
    assert.equal(f.calls.some(([name]) => name === 'dispatch'), false);
    assert.equal(f.calls.some(([name]) => name === 'comment'), false);
  });
}

for (const target of ['all', 'ppu']) {
  test(`allow ${target}`, async () => {
    const f = fixture('admin', `/ci run ${target}`);
    await command(f);
    assert.equal(f.calls.find(([name]) => name === 'dispatch')[1].inputs.platform, target);
  });
}

test('unknown and metadata platforms cannot dispatch', async () => {
  for (const target of ['typo', 'template', 'image_sources']) {
    const f = fixture('write', `/ci run ${target}`);
    await command(f);
    assert.equal(f.calls.some(([name]) => name === 'dispatch'), false);
    assert.equal(f.calls.some(([name]) => name === 'failed'), true);
  }
});

test('closed PR cannot dispatch', async () => {
  const f = fixture();
  f.github.rest.pulls.get = async () => ({ data: { state: 'closed' } });
  await command(f);
  assert.equal(f.calls.some(([name]) => name === 'dispatch'), false);
});

const resolveSource = require('../../.github/scripts/ci/request_source.js');
function requestFixture() {
  const f = fixture();
  const outputs = {};
  f.prNumber = '42';
  f.headSha = f.sha;
  f.core.setOutput = (name, value) => { outputs[name] = value; };
  f.core.summary = {
    addHeading() { return this; }, addRaw() { return this; }, async write() {}
  };
  f.outputs = outputs;
  return f;
}

test('requested workflow resolves source repository through PR API', async () => {
  const f = requestFixture();
  await resolveSource(f);
  assert.deepEqual(f.outputs, { repository: 'contributor/FlagScale', sha: f.sha });
});

test('requested workflow rejects stale SHA', async () => {
  const f = requestFixture();
  f.headSha = 'b'.repeat(40);
  await assert.rejects(resolveSource(f), /head changed/);
  assert.deepEqual(f.outputs, {});
});

test('requested workflow rejects closed or deleted PR sources', async () => {
  for (const pr of [{ state: 'closed', head: {} }, { state: 'open', head: { repo: null } }]) {
    const f = requestFixture();
    f.github.rest.pulls.get = async () => ({ data: pr });
    await assert.rejects(resolveSource(f), /closed, unavailable/);
  }
});

test('requested workflow rejects invalid PR and SHA inputs', async () => {
  for (const [prNumber, headSha] of [['1;echo', 'a'.repeat(40)], ['42', 'main'], ['0', 'a'.repeat(40)]]) {
    const f = requestFixture();
    await assert.rejects(resolveSource({ ...f, prNumber, headSha }), /Expected a PR number/);
  }
});
