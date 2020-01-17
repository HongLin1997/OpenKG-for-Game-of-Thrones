
#    Copyright (C) 2011-2019 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
import json

import flask
import networkx as nx
from networkx.readwrite import json_graph

# Serve the file over http to allow for cross origin requests
app = flask.Flask(__name__, static_folder="static")
app.debug = True

@app.route('/')
def static_proxy():
    return app.send_static_file('graph.html')

print('\nGo to http://localhost:8000 to see the example\n')
app.run(port=8000)