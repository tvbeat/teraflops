#!/usr/bin/env python3

import argparse
import asyncio
import codecs
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

from importlib.resources import files
from string import Template

from rich.progress import (
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
)

from rich.live import Live
from rich.style import Style
from rich.table import Table
from rich.text import Text

TERRAFORM_EXE = 'terraform'
INFO_D = '[[green]INFO [/]]'
WARN_D = '[[yellow]WARN [/]]'
ERROR_D = '[[red]ERROR[/]]'
eval_path = files('teraflops.nix').joinpath('eval.nix')


class TimeElapsedColumn(ProgressColumn):
  """Renders time elapsed."""

  def render(self, task: "Task") -> Text:
    """Show time elapsed."""
    elapsed = task.finished_time if task.finished else task.elapsed
    if elapsed is None:
      return Text('')

    value = max(0, int(elapsed))
    if value > 99:
      m, s = divmod(value, 60)

      value = f'{m}m{s}s' if m > 0 else f'{s}s'
    else:
      value = f'{value}s'

    return Text(value)

# asyncio doesn't have an equivalent exception to subprocess so roll our own
class CalledProcessError(Exception):
  def __init__(self, returncode, stdout=None, stderr=None):
    self.returncode = returncode
    self.stdout = stdout
    self.stderr = stderr


class Console:
  def __init__(self, verbose=False):
    import rich

    self.rich = rich.console.Console(emoji=False)
    self.use_table = verbose

  def print(self, text):
    self.rich.print(text)

  def info(self, text):
    self.rich.print(f'{INFO_D} {text}')

  def warn(self, text):
    self.rich.print(f'{WARN_D} {text}')

  def error(self, text):
    self.rich.print(f'{ERROR_D} {text}')

  @contextlib.contextmanager
  def refresh(self):
    if self.use_table:
      self.table = Table.grid('', '', '') # node, separator, message
      with Live(self.table) as live:
        try:
          yield
        finally:
          live.refresh()
    else:
      with Progress(TextColumn('{task.fields[node]}', justify='right'), SpinnerColumn('clock', style=Style.null(), finished_text=':white_check_mark:'), TimeElapsedColumn(), '{task.description}', console=self.rich) as progress:
        self.progress = progress
        yield


  def message(self, name, description=''):
    if self.use_table:
      assert self.table

      if description != '':
        self.table.add_row(f'[bold]{name}', ' | ', description)
      return name
    else:
      assert self.progress
      return self.progress.add_task(description, total=1, node=f'[bold]{name}')

  def update(self, msg, text, status=None):
    if self.use_table:
      assert self.table
      markup = 'bold'
      if status == 'success':
        markup = markup + ' green'
      if status == 'failure':
        markup = markup + ' red'
      self.table.add_row(f'[{markup}]{msg}', ' | ', text)
    elif text != '':
      assert self.progress
      name = self.progress.tasks[msg].fields['node']

      extra_args = {}
      if status == 'success':
        extra_args = {'advance': 1, 'node': f'[bold green]{name}'}
      if status == 'failure':
        extra_args = {'node': f'[bold red]{name}'}

      self.progress.update(msg, description=text, **extra_args)

##################### pipeline helpers #####################

def ssh_opts(node):
  opts = [
    '-o',
    'StrictHostKeyChecking=accept-new',
    '-o',
    'BatchMode=yes',
    '-T'
  ]

  if os.environ.get('SSH_CONFIG_FILE'):
    opts += ['-F', os.environ['SSH_CONFIG_FILE']]

  if node.get('targetPort'):
    opts += ['-p', node['targetPort']]

  if node.get('targetUser'):
    opts += ['-l', node.get('targetUser')]

  return opts

def ssh_cmd(node, command=None, ssh_args=None):
  cmd = [
    'ssh',
    '-o',
    'StrictHostKeyChecking=accept-new',
    '-o',
    'BatchMode=yes',
  ]

  if command:
    cmd += ['-T']

  if ssh_args:
    cmd += ssh_args

  if os.environ.get('SSH_CONFIG_FILE'):
    cmd += ['-F', os.environ['SSH_CONFIG_FILE']]

  if node.get('targetPort'):
    cmd += ['-p', node['targetPort']]

  if node.get('targetUser'):
    cmd += ['-l', node.get('targetUser')]

  cmd += [node['targetHost']]

  if command:
    cmd += command

  return cmd

async def eval_stage(console, name, terraform_json):
  cmd = [
    'nix-instantiate',
    '--json',
    '--eval',
    eval_path,
    '-A',
    f'nodes."{name}".config.system.build.toplevel.drvPath', '--arg', 'flake', 'builtins.getFlake (toString ./.)', '--argstr', 'terraform_json',
    terraform_json
  ]

  msg = console.message(name, f'evaluating {name}')

  env={}
  process = await asyncio.create_subprocess_exec(*cmd, env={**os.environ, **env}, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

  stderr = b''

  while not process.stderr.at_eof():
    data = await process.stderr.readline()
    stderr += data
    line = data.decode('utf-8').rstrip()

    console.update(msg, line)

  stdout, _ = await process.communicate()

  if process.returncode != 0:
    lines = stderr.decode().rstrip().splitlines()
    if len(lines) > 0:
      console.update(msg, f'evaluation failed: {lines[-1].strip()}', status='failure')
    else:
      console.update(msg, f'evaluation failed: an expected failure occurred', status='failure')

    raise CalledProcessError(process.returncode, stdout=stdout, stderr=stderr)

  toplevel = json.loads(stdout.decode())
  console.update(msg, f'evaluated {toplevel}', status='success')
  return toplevel

async def build_stage(console, name, drv):
  cmd = [
    'nix-build',
    drv,
  ]

  msg = console.message(name, f'building {name}')

  env={}
  process = await asyncio.create_subprocess_exec(*cmd, env={**os.environ, **env}, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

  stderr = b''

  while not process.stderr.at_eof():
    data = await process.stderr.readline()
    stderr += data
    line = data.decode('utf-8').rstrip()

    console.update(msg, line)

  stdout, _ = await process.communicate()

  if process.returncode != 0:
    lines = stderr.decode().rstrip().splitlines()
    if len(lines) > 0:
      console.update(msg, f'build failed: {lines[-1].strip()}', status='failure')
    else:
      console.update(msg, f'build failed: an expected failure occurred', status='failure')

    raise CalledProcessError(process.returncode, stdout=stdout, stderr=stderr)

  toplevel = stdout.decode().strip()
  console.update(msg, f'built {toplevel}', status='success')
  return toplevel

async def copy_stage(console, name, deployment, toplevel):
  cmd = [
    'nix',
    '--extra-experimental-features', 'flakes nix-command',
    'copy',
    '--to', f'ssh-ng://{deployment["targetUser"]}@{deployment["targetHost"]}?compress=true',
    toplevel,
    '--no-check-sigs',
    '--substitute-on-destination',
    '--verbose'
  ]

  msg = console.message(name, 'pushing system closure')

  #env={'NIX_SSHOPTS': ' '.join(ssh_opts(deployment))}
  env={}

  process = await asyncio.create_subprocess_exec(*cmd, env={**os.environ, **env}, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)





  stderr = b''

  while not process.stderr.at_eof():
    data = await process.stderr.readline()
    stderr += data
    line = data.decode('utf-8').rstrip()

    console.update(msg, line)




  await process.wait()

  if process.returncode != 0:
    lines = stderr.decode().rstrip().splitlines()
    # console.info(f'{name} -> len: {len(lines)}, value: "{stderr.decode()}"')
    console.update(msg, f'push failed: {lines[-1].strip()}', status='failure')

    raise CalledProcessError(process.returncode, stdout=stdout, stderr=stderr)

  console.update(msg, 'pushed system closure', status='success')

async def reboot_stage(console, name, node, no_wait=False):
  async def get_boot_id(node):
    ssh_args = ['-o', 'ConnectTimeout=10'] # see https://github.com/zhaofengli/colmena/issues/166#issuecomment-1892325999
    proc = await asyncio.create_subprocess_exec(*ssh_cmd(node, ['cat', '/proc/sys/kernel/random/boot_id'], ssh_args), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    stdout, _ = await proc.communicate()

    return None if proc.returncode != 0 else stdout.decode()

  async def initiate_reboot(node):
    proc = await asyncio.create_subprocess_exec(*ssh_cmd(node, ['reboot']), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    stdout, _ = await proc.communicate()

    if proc.returncode == 0 or proc.returncode == 255:
      return stdout.decode()

  msg = console.message(name)

  console.update(msg, 'rebooting')

  if no_wait:
    return await initiate_reboot(node)

  old_id = await get_boot_id(node)

  await initiate_reboot(node)

  console.update(msg, 'waiting for reboot')

  while True:
    new_id = await get_boot_id(node)
    if new_id and new_id != old_id:
      break

    await asyncio.sleep(2)

  console.update(msg, 'rebooted', status='success')

async def upload_keys_stage(console, name, deployment, terraform_json):
  cmd = [
    'nix',
    '--extra-experimental-features', 'flakes nix-command',
    'eval',
    '--impure',
    '--json',
    '--expr',
    f'(import {eval_path} {{ flake = builtins.getFlake (toString ./.); terraform_json = {terraform_json}; }}).nodes."{name}".config.deployment.keys'
  ]

  msg = console.message(name, 'uploading keys')

  process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
  stdout, stderr = await process.communicate()

  if process.returncode != 0:
    console.update(msg, f'key upload failed: {stderr.decode().strip()}', status='failure')
    raise CalledProcessError(process.returncode, stdout=stdout, stderr=stderr)

  data = json.loads(stdout)

  for key in data.values():
    console.update(msg, f'uploading {key["name"]}')
    value = Template(files('teraflops').joinpath('key_uploader.template.sh').read_text()).safe_substitute(DESTINATION=key['path'], USER=key['user'], GROUP=key['group'], PERMISSIONS=key['permissions'], REQUIRE_OWNERSHIP='1')

    process = await asyncio.create_subprocess_exec(*ssh_cmd(deployment, ['sh', '-c', value]), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, stdin=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate(key['text'].encode())

    if process.returncode != 0:
      console.update(msg, f'key upload failed: {stderr}', status='failure')
      raise CalledProcessError(process.returncode, stdout=stdout, stderr=stderr)

  console.update(msg, 'uploaded keys', status='success')

async def switch_to_configuration_stage(console, name, node, toplevel, target):
  msg = console.message(name, 'activating system profile')

  cmd = ssh_cmd(node, [f'{toplevel}/bin/switch-to-configuration', target])

  process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

  stdout = b''

  while not process.stdout.at_eof():
    data = await process.stdout.readline()
    stdout += data
    line = data.decode('utf-8').rstrip()

    console.update(msg, line)

  _, stderr = await process.communicate()




  if process.returncode != 0: # done, error
    console.update(msg, f'activation failed: {stderr.decode().strip()}', status='failure')
    raise CalledProcessError(returncode, stdout=stdout, stderr=stderr)

  console.update(msg, 'activation successful', status='success')

async def wait_for_node(console, name, node):
  msg = None

  while True:
    # see https://github.com/zhaofengli/colmena/issues/166#issuecomment-1892325999
    proc = await asyncio.create_subprocess_exec(*ssh_cmd(node, ['cat', '/proc/sys/kernel/random/boot_id'], ['-o', 'ConnectTimeout=10']), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)

    if await proc.wait() == 0:
      break

    if not msg:
      msg = console.message(name, 'waiting for node to become available')

    await asyncio.sleep(2)
  
  if msg:
    console.update(msg, description='node is now available', status='success')

############################################################






async def filter_nodes(args):
  all_node_names = await get_nodes_names()

  # TODO: is querying tags from nix significantly slower than querying from terraform??
  async def get_node_tags(name):
    cmd_args = [
      '--extra-experimental-features', 'flakes nix-command',
      'eval',
      '--impure',
      '--json',
      '--expr',
      f'(import {eval_path} {{ flake = builtins.getFlake (toString ./.); }}).nodes."{name}".config.deployment.tags'
    ]

    process = await asyncio.create_subprocess_exec('nix', *cmd_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
      raise CalledProcessError(process.returncode, stdout, stderr)

    return name, json.loads(stdout)

  #########################################

  on = args.on.split(',') if args.on else None

  if on is None:
    return all_node_names, all_node_names

  tags_to_check = set([value[1:] for value in on if value.startswith('@')])
  names_to_check = [value for value in on if not value.startswith('@')]

  if tags_to_check:
    async with asyncio.TaskGroup() as tg:
      tasks = [tg.create_task(get_node_tags(name)) for name in all_node_names]

    tag_map = dict(task.result() for task in tasks)

    return [name for name in all_node_names if (name in names_to_check) or (  not tags_to_check.isdisjoint(tag_map[name])  )], all_node_names

  return [name for name in all_node_names if name in names_to_check], all_node_names






async def get_nodes_names():
  cmd = [
    'nix',
    '--extra-experimental-features', 'flakes nix-command',
    'eval',
    '--impure',
    '--json',
    '--expr',
    f'builtins.attrNames (import {eval_path} {{ flake = builtins.getFlake (toString ./.); }}).nodes'
  ]

  process = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
  stdout, stderr = await process.communicate()

  if process.returncode != 0:
    raise CalledProcessError(process.returncode, stdout, stderr)

  return json.loads(stdout)


@contextlib.asynccontextmanager
async def generate_full_terraform_config():
  process = await asyncio.create_subprocess_exec('nix-build', '--quiet', '--no-out-link', eval_path, '-A', 'terraform', '--arg', 'flake', 'builtins.getFlake (toString ./.)', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
  stdout, stderr = await process.communicate()

  if process.returncode != 0:
    raise CalledProcessError(process.returncode, stdout, stderr)

  tf_json = stdout.strip()

  shutil.copy(tf_json, 'main.tf.json')
  os.chmod('main.tf.json', 0o664)

  try:
    yield
  finally:
    with contextlib.suppress(FileNotFoundError):
      os.remove('main.tf.json')


@contextlib.asynccontextmanager
async def generate_minimal_terraform_config(use_cache_if_available = True):
  tf_data_dir = os.getenv('TF_DATA_DIR', '.terraform')
  tf_cache_file = os.path.join(tf_data_dir, 'teraflops.json')

  if not os.path.isfile(tf_cache_file) or not use_cache_if_available:
    process = await asyncio.create_subprocess_exec('nix-build', '--quiet', '--no-out-link', eval_path, '-A', 'bootstrap', '--arg', 'flake', 'builtins.getFlake (toString ./.)', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
      raise CalledProcessError(process.returncode, stdout, stderr)

    tf_json = stdout.strip()

    os.makedirs(tf_data_dir, exist_ok=True)
    shutil.copy(tf_json, tf_cache_file)
    os.chmod(tf_cache_file, 0o664)

  shutil.copy(tf_cache_file, 'main.tf.json')

  try:
    yield
  finally:
    with contextlib.suppress(FileNotFoundError):
      os.remove('main.tf.json')


@contextlib.asynccontextmanager
async def generate_terraform_data_for_nix():

  async with generate_minimal_terraform_config():
    process = await asyncio.create_subprocess_exec(TERRAFORM_EXE, 'show', '-json', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
      raise CalledProcessError(process.returncode, stdout, stderr)

  terraform_data = json.loads(stdout)
  try:
    outputs = terraform_data['values']['outputs']
    resources = terraform_data['values']['root_module']['resources']
  except KeyError:
    resources = dict()
    outputs = dict()

  resources_data = dict()
  for resource in resources:
    inner = resources_data.setdefault(resource['type'], dict())

    if resource.get('index') is not None:
      if type(resource.get('index')) == int:
        offset = int(resource.get('index'))
        index = inner.setdefault(resource['name'], list())
        index += [None] * ((offset + 1) - len(index))
        index.insert(offset, resource['values'])
      else:
        index = inner.setdefault(resource['name'], dict())
        index[resource['index']] = resource['values']
    else:
      inner[resource['name']] = resource['values']

  outputs_data = dict()
  for key, value in outputs.items():
    outputs_data[key] = value['value']

  with tempfile.TemporaryDirectory(prefix='teraflops-rewrite.', delete=True) as tempdir:
    terraform_json = os.path.join(tempdir, 'terraform.json')
    with open(terraform_json, 'w') as f:
      f.write(json.dumps(dict(outputs=outputs_data, resources=resources_data)))
      f.close()

      yield terraform_json






async def get_teraflops_data():
  async with generate_minimal_terraform_config():
    process = await asyncio.create_subprocess_exec(TERRAFORM_EXE, 'output', '-json', 'teraflops', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
      raise CalledProcessError(process.returncode, stdout, stderr)

    return json.loads(stdout)
























async def init(args):
  cmd = [TERRAFORM_EXE, 'init']
  if args.migrate_state:
    cmd += ['-migrate-state']
  if args.reconfigure:
    cmd += ['-reconfigure']
  if args.upgrade:
    cmd += ['-upgrade']

  async with generate_minimal_terraform_config(use_cache_if_available=False):
    subprocess.run(cmd, check=True)





async def repl(args):
  async with generate_terraform_data_for_nix() as terraform_json:
    cmd = [
      'nix',
      'repl',
      '--experimental-features', 'flakes nix-command',
      '--expr', f'with import {eval_path} {{ flake = builtins.getFlake (toString ./.); terraform_json = {terraform_json}; }}; repl'
    ]
    if args.debugger:
      cmd += ['--debugger']

    subprocess.run(cmd, check=True)




async def eval(args):
  console = Console(args.verbose)

  async def nix_eval_expr(nix_expr, terraform_json):
    cmd_args = [
      '--extra-experimental-features', 'flakes nix-command',
      'eval',
      '--impure',
      '--expr',
      f'(import {eval_path} {{ flake = builtins.getFlake (toString ./.); terraform_json = {terraform_json or "null"}; }}).evalFn ({nix_expr})'
    ]

    if args.json:
      cmd_args += ['--json']
    if args.raw:
      cmd_args += ['--raw']

    process = await asyncio.create_subprocess_exec('nix', *cmd_args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()
    
    return process.returncode, stdout, stderr

  context = contextlib.nullcontext() if args.without_resources else generate_terraform_data_for_nix()
  async with context as terraform_json:
    tasks = [nix_eval_expr(expr, terraform_json) for expr in args.expr]
    for task in asyncio.as_completed(tasks):
      returncode, stdout, stderr = await task

      if returncode != 0:
        raise CalledProcessError(returncode, stdout=stdout, stderr=stderr)

      console.print(stdout.strip().decode())




async def plan(args):
  async with generate_full_terraform_config():
    subprocess.run([TERRAFORM_EXE, 'plan'])



async def apply(args):
  cmd = [TERRAFORM_EXE, 'apply']
  if args.confirm:
    cmd += ['-auto-approve']

  async with generate_full_terraform_config():
    subprocess.run(cmd)



async def build(args):
  errors = {}
  console = Console(args.verbose)

  async def pipeline(console, name, terraform_json, drv):
    try:
      if drv is None:
        drv = await eval_stage(console, name, terraform_json)
      await build_stage(console, name, drv)
    except CalledProcessError as e:
      errors[name] = e

  console.info('enumerating nodes...')
  selected, all = await filter_nodes(args)
  if len(all) == len(selected):
    console.info(f'selected all {len(selected)} nodes')
  else:
    console.info(f'selected {len(selected)} out of {len(all)} hosts')

  with console.refresh():

    context = contextlib.nullcontext() if args.with_drvs else generate_terraform_data_for_nix()
    async with context as terraform_json:
      if not args.with_drvs:
        console.info('terraform data gathered, ready to do work')

      if args.with_drvs:
        with open(args.with_drvs) as f:
          drvs = json.load(f)

      async with asyncio.TaskGroup() as tg:
        for name in selected:
          tg.create_task(pipeline(console, name, terraform_json, drvs[name] if args.with_drvs else None))

  for name, e in errors.items():
    console.error(f'failed to build to {name} - logs:')

    # https://stackoverflow.com/a/37059682
    value = codecs.escape_decode(e.stderr)[0].decode('utf-8')
    for line in value.splitlines():
      console.error(f'  stderr) {line}')

    console.error(f' failure) child process exited with error code: {e.returncode}')

  if len(errors) != 0:
    sys.exit(1)





async def push(args):
  errors = {}
  console = Console(args.verbose)

  async def pipeline(console, name, deployment, terraform_json, drv):
    try:
      if drv is None:
        drv = await eval_stage(console, name, terraform_json)
      toplevel = await build_stage(console, name, drv)
      await copy_stage(console, name, deployment, toplevel)
    except Exception as e:
      errors[name] = e

  console.info('enumerating nodes...')
  selected, all = await filter_nodes(args)
  if len(all) == len(selected):
    console.info(f'selected all {len(selected)} nodes')
  else:
    console.info(f'selected {len(selected)} out of {len(all)} hosts')

  output_data = await get_teraflops_data()

  console.info('teraflops data gathered')

  with console.refresh():

    context = contextlib.nullcontext() if args.with_drvs else generate_terraform_data_for_nix()
    async with context as terraform_json:
      if not args.with_drvs:
        console.info('terraform data gathered, ready to do work')

      if args.with_drvs:
        with open(args.with_drvs) as f:
          drvs = json.load(f)

      async with asyncio.TaskGroup() as tg:
        for name, deployment in output_data['nodes'].items():
          if name in selected:
            tg.create_task(pipeline(console, name, deployment, terraform_json, drvs[name] if args.with_drvs else None))

  for name, e in errors.items():
    if not hasattr(e, 'stderr'):
      console.error('EXCEPTION HAS NO STDERR')
      console.error(e)
    #  continue

    console.error(f'failed to push to {name} - logs:')

    # https://stackoverflow.com/a/37059682
    value = codecs.escape_decode(e.stderr)[0].decode('utf-8')
    for line in value.splitlines():
      console.error(f'  stderr) {line}')

    console.error(f' failure) child process exited with error code: {e.returncode}')

  if len(errors) != 0:
    sys.exit(1)



async def upload_keys(args):
  errors = {}
  console = Console(args.verbose)

  async def pipeline(console, name, node, terraform_json):
    try:
      await upload_keys_stage(console, name, node, terraform_json)
    except CalledProcessError as e:
      errors[name] = e

  console.info('enumerating nodes...')
  selected, all = await filter_nodes(args)
  if len(all) == len(selected):
    console.info(f'selected all {len(selected)} nodes')
  else:
    console.info(f'selected {len(selected)} out of {len(all)} hosts')

  with console.refresh():
    output_data = await get_teraflops_data()
    console.info('teraflops data gathered')

    async with generate_terraform_data_for_nix() as terraform_json:
      console.info('terraform data gathered, ready to do work')
      async with asyncio.TaskGroup() as tg:
        for name in selected:
          tg.create_task(pipeline(console, name, output_data['nodes'][name], terraform_json))

  for name, e in errors.items():
    console.error(f'failed to upload keys to {name} - logs:')

    # https://stackoverflow.com/a/37059682
    value = codecs.escape_decode(e.stderr)[0].decode('utf-8')
    for line in value.splitlines():
      console.error(f'  stderr) {line}')

    console.error(f' failure) child process exited with error code: {e.returncode}')

  if len(errors) != 0:
    sys.exit(1)



async def activate(args):
  errors = {}
  console = Console(args.verbose)

  async def pipeline(console, name, deployment, terraform_json, drv):
    try:
      if drv is None:
        drv = await eval_stage(console, name, terraform_json)
      toplevel = await build_stage(console, name, drv)
      await copy_stage(console, name, deployment, toplevel)

      if not args.no_keys and not args.dry_run:
        await upload_keys_stage(console, name, deployment, terraform_json)

      if args.reboot:
        target = 'boot'
      elif args.dry_run:
        target = 'dry-activate'
      else:
        target = 'switch'

      await switch_to_configuration_stage(console, name, deployment, toplevel, target)

      # TODO: upload post activation keys

      if args.reboot:
        value = await reboot_stage(console, name, deployment)
    except CalledProcessError as e:
      errors[name] = e

  console.info('enumerating nodes...')
  selected, all = await filter_nodes(args)
  if len(all) == len(selected):
    console.info(f'selected all {len(selected)} nodes')
  else:
    console.info(f'selected {len(selected)} out of {len(all)} hosts')

  with console.refresh():
    output_data = await get_teraflops_data()

    console.info('teraflops data gathered')

    async with generate_terraform_data_for_nix() as terraform_json:
      console.info('terraform data gathered, ready to do work')

      if args.with_drvs:
        with open(args.with_drvs) as f:
          drvs = json.load(f)

      async with asyncio.TaskGroup() as tg:
        for name, deployment in output_data['nodes'].items():
          if name in selected:
            tg.create_task(pipeline(console, name, deployment, terraform_json, drvs[name] if args.with_drvs else None))

  for name, e in errors.items():
    console.error(f'failed to activate {name} - logs:')

    # https://stackoverflow.com/a/37059682
    value = codecs.escape_decode(e.stderr)[0].decode('utf-8')
    for line in value.splitlines():
      console.error(f'  stderr) {line}')

    console.error(f' failure) child process exited with error code: {e.returncode}')

  if len(errors) != 0:
    sys.exit(1)


async def deploy(args):
  errors = {}
  console = Console(args.verbose)

  async def pipeline(console, name, deployment, terraform_json):
    try:
      await wait_for_node(console, name, node)

      drv = await eval_stage(console, name, terraform_json)
      toplevel = await build_stage(console, name, drv)
      await copy_stage(console, name, deployment, toplevel)

      if not args.no_keys:
        await upload_keys_stage(console, name, deployment, terraform_json)

      await switch_to_configuration_stage(console, name, deployment, toplevel, 'boot' if args.reboot else 'switch')

      # TODO: upload post activation keys

      if args.reboot:
        value = await reboot_stage(console, name, deployment)
        # console.info(f'{name} got return from reboot_stage: {value}')
    except CalledProcessError as e:
      errors[name] = e

  # apply terraform configuration
  cmd = [TERRAFORM_EXE, 'apply']
  if args.confirm:
    cmd += ['-auto-approve']

  async with generate_full_terraform_config():
    subprocess.run(cmd)
  # apply terraform configuration

  console.info('enumerating nodes...')
  selected, all = await filter_nodes(args)
  if len(all) == len(selected):
    console.info(f'selected all {len(selected)} nodes')
  else:
    console.info(f'selected {len(selected)} out of {len(all)} hosts')

  with console.refresh():
    output_data = await get_teraflops_data()

    console.info('teraflops data gathered')

    async with generate_terraform_data_for_nix() as terraform_json:
      console.info('terraform data gathered, ready to do work')
      async with asyncio.TaskGroup() as tg:
        for name, node in output_data['nodes'].items():
          if name in selected:
            tg.create_task(pipeline(console, name, node, terraform_json))

  for name, e in errors.items():
    console.error(f'failed to deploy {name} - logs:')

    # https://stackoverflow.com/a/37059682
    value = codecs.escape_decode(e.stderr)[0].decode('utf-8')
    for line in value.splitlines():
      console.error(f'  stderr) {line}')

    console.error(f' failure) child process exited with error code: {e.returncode}')

  if len(errors) != 0:
    sys.exit(1)

async def destroy(args):
  cmd = [TERRAFORM_EXE, 'apply', '-destroy']
  if args.confirm:
    cmd += ['-auto-approve']

  async with generate_full_terraform_config():
    subprocess.run(cmd)






async def check(args):
  console = Console(args.verbose)
  async def doit(console, name, node, command):
    # TODO: too verbose?
    msg = console.message(name, f'executing {command} on {node["targetHost"]}')

    process = await asyncio.create_subprocess_exec(*ssh_cmd(node, ['uptime']), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    stdout, _ = await process.communicate()

    if process.returncode != 0:
      console.update(msg, 'unavailable', status='failure')
    else:
      console.update(msg, stdout.decode().strip(), status='success')

  output_data = await get_teraflops_data()

  with console.refresh():
    async with asyncio.TaskGroup() as tg:
      for name, deployment in output_data['nodes'].items():
        tg.create_task(doit(console, name, deployment, ['uptime']))



async def ssh(args):
  output_data = await get_teraflops_data()

  node = output_data['nodes'][args.node]

  subprocess.run(ssh_cmd(node))





async def ssh_for_each(args):
  console = Console(args.verbose)

  async def doit(console, name, node, command):
    msg = console.message(name)

    if args.verbose:
      console.update(msg, 'executing remote command')

    process = await asyncio.create_subprocess_exec(*ssh_cmd(node, command), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    stdout, stderr = await process.communicate()

    if process.returncode != 0:
      console.update(msg, f'failed: {stderr.decode().strip()}', status='failure')
    else:
      console.update(msg, stdout.decode().strip(), status='success')

  output_data = await get_teraflops_data()

  console.info('enumerating nodes...')
  selected, all = await filter_nodes(args)
  if len(all) == len(selected):
    console.info(f'selected all {len(selected)} nodes')
  else:
    console.info(f'selected {len(selected)} out of {len(all)} hosts')

  with console.refresh():

    async with asyncio.TaskGroup() as tg:
      for name in selected:
        tg.create_task(doit(console, name, output_data['nodes'][name], args.command))














async def reboot(args):
  console = Console(verbose=True)

  # TODO: some sort of message, right?
  output_data = await get_teraflops_data()

  console.info('enumerating nodes...')
  selected, all = await filter_nodes(args)
  if len(all) == len(selected):
    console.info(f'selected all {len(selected)} nodes')
  else:
    console.info(f'selected {len(selected)} out of {len(all)} hosts')

  with console.refresh():
    async with asyncio.TaskGroup() as tg:
      for name in selected:
        tg.create_task(reboot_stage(console, name, output_data['nodes'][name], args.no_wait))



async def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('-v', '--verbose', action='store_true')

  confirm_parser = argparse.ArgumentParser(add_help=False)
  confirm_parser.add_argument('--confirm', action='store_true', help='confirm dangerous operations; do not ask')

  on_parser = argparse.ArgumentParser(add_help=False)
  on_parser.add_argument('--on', metavar='<NODES>', help='select a list of nodes to deploy to')

  no_keys_parser = argparse.ArgumentParser(add_help=False)
  no_keys_parser.add_argument('--no-keys', action='store_true', help='do not upload secret keys set in `deployment.keys`')

  with_drvs = argparse.ArgumentParser(add_help=False)
  with_drvs.add_argument('--with-drvs', metavar='<FILE>', help='a file including json key value pairs of node name and the associated top level evaluated drv')

  subparsers = parser.add_subparsers(title='subcommands')

  init_parser = subparsers.add_parser('init', help='prepare your working directory for other commands')
  init_parser.set_defaults(func=init)
  init_parser.add_argument('--migrate-state', action='store_true', help='reconfigure a backend, and attempt to migrate any existing state')
  init_parser.add_argument('--reconfigure', action='store_true', help='reconfigure a backend, ignoring any saved configuration')
  init_parser.add_argument('--upgrade', action='store_true', help='install the latest module and provider versions allowed within configured constraints, overriding the default behavior of selecting exactly the version recorded in the dependency lockfile.')

  repl_parser = subparsers.add_parser('repl', help='start an interactive REPL with the complete configuration')
  repl_parser.set_defaults(func=repl)
  repl_parser.add_argument('--debugger', action='store_true', help='start an interactive environment if evaluation fails')

  eval_parser = subparsers.add_parser('eval', help='evaluate an expression using the complete configuration')
  eval_parser.set_defaults(func=eval)
  eval_parser.add_argument('expr', type=str, nargs='+', help='the nix expression') # TODO: add an example, { resources, nodes, pkgs, lib, ... }:
  eval_parser.add_argument('--without-resources', action='store_true')
  group = eval_parser.add_mutually_exclusive_group()
  group.add_argument('--json', action='store_true')
  group.add_argument('--raw', action='store_true')

  plan_parser = subparsers.add_parser('plan', help='show changes required by the current configuration')
  plan_parser.set_defaults(func=plan)

  apply_parser = subparsers.add_parser('apply', parents=[confirm_parser], help='create or update all resources in the deployment')
  apply_parser.set_defaults(func=apply)

  build_parser = subparsers.add_parser('build', parents=[on_parser, with_drvs], help='build the system profiles')
  build_parser.set_defaults(func=build)

  push_parser = subparsers.add_parser('push', parents=[on_parser, with_drvs], help='copy the closures to remote nodes')
  push_parser.set_defaults(func=push)

  upload_keys_parser = subparsers.add_parser('upload-keys', parents=[on_parser], help='upload keys to remote hosts')
  upload_keys_parser.set_defaults(func=upload_keys)

  activate_parser = subparsers.add_parser('activate', parents=[on_parser, no_keys_parser, with_drvs], help='apply configurations on remote nodes')
  activate_parser.set_defaults(func=activate)
  group = activate_parser.add_mutually_exclusive_group()
  group.add_argument('--reboot', action='store_true', help='reboots nodes after activation and waits for them to come back up')
  group.add_argument('--dry-run', action='store_true', help='show what changes would be performed by the activation')

  deploy_parser = subparsers.add_parser('deploy', parents=[confirm_parser, on_parser, no_keys_parser], help='deploy the configuration')
  deploy_parser.set_defaults(func=deploy)
  deploy_parser.add_argument('--reboot', action='store_true', help='reboots nodes after activation and waits for them to come back up')

  destroy_parser = subparsers.add_parser('destroy', parents=[confirm_parser], help='destroy all resources in the deployment')
  destroy_parser.set_defaults(func=destroy)


  check_parser = subparsers.add_parser('check', help='attempt to connect to each node via SSH and print the results of the uptime command.')
  check_parser.set_defaults(func=check)

  ssh_parser = subparsers.add_parser('ssh', help='login on the specified machine via SSH')
  ssh_parser.set_defaults(func=ssh)
  ssh_parser.add_argument('node', type=str, help='identifier of the node')

  ssh_for_each_parser = subparsers.add_parser('ssh-for-each', parents=[on_parser], help='execute a command on each machine via SSH')
  ssh_for_each_parser.set_defaults(func=ssh_for_each)
  ssh_for_each_parser.add_argument('command', nargs=argparse.REMAINDER, help='command to run')



  reboot_parser = subparsers.add_parser('reboot', parents=[on_parser], help='reboot all nodes in the deployment')
  reboot_parser.set_defaults(func=reboot)
  reboot_parser.add_argument('--no-wait', action='store_true', help='do not wait until the nodes are up again')

  args = parser.parse_args()

  # call the appropriate function based on the subcommand
  if hasattr(args, 'func'):
    try:
      await args.func(args)
    except CalledProcessError as e:
      console = Console(args.verbose)

      # https://stackoverflow.com/a/37059682
      value = codecs.escape_decode(e.stderr)[0].decode('utf-8')
      for line in value.splitlines():
        console.error(f'  stderr) {line}')
  else:
    # if no subcommand is provided, print help
    parser.print_help()

if __name__ == '__main__':
  run()

def run():
  asyncio.run(main())
