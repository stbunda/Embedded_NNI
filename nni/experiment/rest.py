# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Any, Optional

import requests

# API URL must be synchronized with:
#  - ts/nni_manager/rest_server/index.ts
#  - ts/webui/src/static/constant.ts
# Remember to update them if the values are changed, or if this file is moved.

_logger = logging.getLogger(__name__)

timeout = 200

def request(method: str, port: Optional[int], api: str, data: Any = None, prefix: Optional[str] = None, node: Optional[str] = None) -> Any:
    if port is None:
        raise RuntimeError('Experiment is not running')
    print('node:', node)

    if node is not None:
        url_parts = [
            f'http://{node}:{port}',
            prefix,
            'api/v1/nni',
            api
        ]
    else:
        url_parts = [
            f'http://localhost:{port}',
            prefix,
            'api/v1/nni',
            api
        ]
    url = '/'.join(part.strip('/') for part in url_parts if part)
    print(url)

    if data is None:
        print('data is None')
        resp = requests.request(method, url, timeout=timeout)
    else:
        resp = requests.request(method, url, json=data, timeout=timeout)
        print('resp', resp)

    if not resp.ok:
        _logger.error('rest request %s %s failed: %s %s', method.upper(), url, resp.status_code, resp.text)
    resp.raise_for_status()

    if method.lower() in ['get', 'post'] and len(resp.content) > 0:
        return resp.json()

def get(port: Optional[int], api: str, prefix: Optional[str] = None, node: Optional[str] = None) -> Any:
    return request('get', port, api, prefix=prefix, node=node)

def post(port: Optional[int], api: str, data: Any, prefix: Optional[str] = None, node: Optional[str] = None) -> Any:
    return request('post', port, api, data, prefix=prefix, node=node)

def put(port: Optional[int], api: str, data: Any, prefix: Optional[str] = None, node: Optional[str] = None) -> None:
    request('put', port, api, data, prefix=prefix, node=node)

def delete(port: Optional[int], api: str, prefix: Optional[str] = None, node: Optional[str] = None) -> None:
    request('delete', port, api, prefix=prefix, node=node)
