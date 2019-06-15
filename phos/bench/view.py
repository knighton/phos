#!/usr/bin/env python3

from argparse import ArgumentParser
from flask import Flask, jsonify, request, Response
from mimetypes import guess_type

from .lookup import get_blurb, get_done_models, get_settings, \
    get_static_query_options, query_stats


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--host', type=str, default='0.0.0.0')
    a.add_argument('--port', type=int, default=1337)
    a.add_argument('--dir', type=str, default='data/bench/example/')
    a.add_argument('--inc_dir', type=str, default='phos/bench/www/inc')
    a.add_argument('--template_dir', type=str, default='phos/bench/www/template')
    return a.parse_args()


flags = parse_flags()

app = Flask(__name__)


@app.route('/api/get_settings', methods=['POST'])
def serve_api_get_settings():
    x = get_settings(flags.dir)
    return jsonify(x)


@app.route('/api/get_done_models', methods=['POST'])
def serve_api_get_done_models():
    x = get_done_models(flags.dir)
    return jsonify(x)


@app.route('/api/get_static_query_options', methods=['POST'])
def serve_api_get_static_query_options():
    x = get_static_query_options()
    return jsonfiy(x)


@app.route('/api/query_results', methods=['POST'])
def serve_api_query_results():
    x = request.get_json(force=True)
    x = query_stats(flags.dir, x)
    return jsonify(x)


@app.route('/inc/<basename>')
def serve_inc(basename):
    assert '..' not in basename
    f = '%s/%s' % (flags.inc_dir, basename)
    text = open(f, 'rb').read()
    mimetype = guess_type(basename)[0]
    return Response(text, mimetype=mimetype)


@app.route('/')
def serve():
    f = '%s/index.html' % flags.template_dir
    return open(f).read()


app.run(host=flags.host, port=flags.port)
