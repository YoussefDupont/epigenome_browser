function neighbourhoodHighlight(params) {
  // console.log("in nieghbourhoodhighlight");
  allNodes = nodes.get({ returnType: "Object" });
  // Strip initial layout coordinates so update() doesn't snap nodes back
  for (var _nid in allNodes) { delete allNodes[_nid].x; delete allNodes[_nid].y; }
  // originalNodes = JSON.parse(JSON.stringify(allNodes));
  // if something is selected:
  if (params.nodes.length > 0) {
    highlightActive = true;
    var i, j;
    var selectedNode = params.nodes[0];
    
    // Check if the selected node exists (skip cluster nodes)
    if (!allNodes[selectedNode]) {
      return;
    }
    
    var degrees = 2;

    // Mark all nodes as hard to read
    for (let nodeId in allNodes) {
      // nodeColors[nodeId] = allNodes[nodeId].color;
      allNodes[nodeId].color = "rgba(200,200,200,0.5)";
      if (allNodes[nodeId].hiddenLabel === undefined) {
        allNodes[nodeId].hiddenLabel = allNodes[nodeId].label;
        allNodes[nodeId].label = undefined;
      }
    }
    var connectedNodes = network.getConnectedNodes(selectedNode);
    var allConnectedNodes = [];

    // Get the second degree nodes
    for (i = 1; i < degrees; i++) {
      for (j = 0; j < connectedNodes.length; j++) {
        allConnectedNodes = allConnectedNodes.concat(
          network.getConnectedNodes(connectedNodes[j])
        );
      }
    }

    // All second degree nodes get a different color (labels stay hidden)
    for (i = 0; i < allConnectedNodes.length; i++) {
      if (allNodes[allConnectedNodes[i]]) {
        allNodes[allConnectedNodes[i]].color = "rgba(150,150,150,0.75)";
      }
    }

    // All first degree nodes get their own color and their label back
    for (i = 0; i < connectedNodes.length; i++) {
      if (allNodes[connectedNodes[i]]) {
        // allNodes[connectedNodes[i]].color = undefined;
        allNodes[connectedNodes[i]].color = nodeColors[connectedNodes[i]];
        if (allNodes[connectedNodes[i]].hiddenLabel !== undefined) {
          allNodes[connectedNodes[i]].label =
            allNodes[connectedNodes[i]].hiddenLabel;
          allNodes[connectedNodes[i]].hiddenLabel = undefined;
        }
      }
    }

    // the main node gets its selected color and its label back
    allNodes[selectedNode].color = allNodes[selectedNode].lightColour || nodeColors[selectedNode];
    if (allNodes[selectedNode].hiddenLabel !== undefined) {
      allNodes[selectedNode].label = allNodes[selectedNode].hiddenLabel;
      allNodes[selectedNode].hiddenLabel = undefined;
    }
  } else if (highlightActive === true) {
    // reset all nodes - re-hide any labels that were made visible during highlight
    for (let nodeId in allNodes) {
      allNodes[nodeId].color = nodeColors[nodeId];
      if (allNodes[nodeId].label !== undefined) {
        allNodes[nodeId].hiddenLabel = allNodes[nodeId].label;
        allNodes[nodeId].label = undefined;
      }
    }
    highlightActive = false;
    window._intervalHighlightedIds = []; 
  }

  // Transform the object into an array
  var updateArray = [];
  if (params.nodes.length > 0) {
    for (let nodeId in allNodes) {
      if (allNodes.hasOwnProperty(nodeId)) {
        // console.log(allNodes[nodeId]);
        updateArray.push(allNodes[nodeId]);
      }
    }
    nodes.update(updateArray);
    // Zoom to fit the selected node and its direct neighbours
    network.fit({
      nodes: [selectedNode].concat(connectedNodes),
      animation: { duration: 500, easingFunction: 'easeInOutQuad' }
    });
  } else {
    // console.log("Nothing was selected");
    for (let nodeId in allNodes) {
      if (allNodes.hasOwnProperty(nodeId)) {
        updateArray.push(allNodes[nodeId]);
      }
    }
    nodes.update(updateArray);
    // Zoom back out to the full graph
    network.fit({ animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
  }
}

function filterHighlight(params) {
  allNodes = nodes.get({ returnType: "Object" });
  allEdges = edges.get({ returnType: "Object" });
  // Strip initial layout coordinates so update() doesn't snap nodes back
  for (var _nid in allNodes) { delete allNodes[_nid].x; delete allNodes[_nid].y; }
  var selectedNodes = params.nodes || [];
  var selectedEdges = params.edges || [];
  // if something is selected:
  if (selectedNodes.length > 0 || selectedEdges.length > 0) {
    filterActive = true;

    // Hiding all nodes and saving the label
    for (let nodeId in allNodes) {
      allNodes[nodeId].hidden = true;
      if (allNodes[nodeId].savedLabel === undefined) {
        allNodes[nodeId].savedLabel = allNodes[nodeId].label;
        allNodes[nodeId].label = undefined;
      }
    }

    for (let i=0; i < selectedNodes.length; i++) {
      if (allNodes[selectedNodes[i]]) {
        allNodes[selectedNodes[i]].hidden = false;
        if (allNodes[selectedNodes[i]].savedLabel !== undefined) {
          allNodes[selectedNodes[i]].label = allNodes[selectedNodes[i]].savedLabel;
          allNodes[selectedNodes[i]].savedLabel = undefined;
        }
      }
    }

    // Hide all edges first
    for (let edgeId in allEdges) {
      allEdges[edgeId].hidden = true;
    }

    // Show selected edges and their endpoint nodes
    for (let i = 0; i < selectedEdges.length; i++) {
      var eid = selectedEdges[i];
      var edge = allEdges[eid];
      if (!edge) continue;
      edge.hidden = false;
      var fromId = edge.from;
      var toId = edge.to;
      if (allNodes[fromId]) {
        allNodes[fromId].hidden = false;
        if (allNodes[fromId].savedLabel !== undefined) {
          allNodes[fromId].label = allNodes[fromId].savedLabel;
          allNodes[fromId].savedLabel = undefined;
        }
      }
      if (allNodes[toId]) {
        allNodes[toId].hidden = false;
        if (allNodes[toId].savedLabel !== undefined) {
          allNodes[toId].label = allNodes[toId].savedLabel;
          allNodes[toId].savedLabel = undefined;
        }
      }
    }

    // If nodes were selected directly, keep edges connecting those visible nodes
    if (selectedNodes.length > 0 && selectedEdges.length === 0) {
      for (let edgeId in allEdges) {
        var e = allEdges[edgeId];
        if (allNodes[e.from] && allNodes[e.to] && !allNodes[e.from].hidden && !allNodes[e.to].hidden) {
          e.hidden = false;
        }
      }
    }

  } else if (filterActive === true) {
    // reset all nodes
    for (let nodeId in allNodes) {
      allNodes[nodeId].hidden = false;
      if (allNodes[nodeId].savedLabel !== undefined) {
        allNodes[nodeId].label = allNodes[nodeId].savedLabel;
        allNodes[nodeId].savedLabel = undefined;
      }
    }

    // Reset all edges
    for (let edgeId in allEdges) {
      allEdges[edgeId].hidden = false;
    }
    filterActive = false;
  }

  // Transform the object into an array
  var updateArray = [];
  if (params.nodes.length > 0) {
    for (let nodeId in allNodes) {
      if (allNodes.hasOwnProperty(nodeId)) {
        updateArray.push(allNodes[nodeId]);
      }
    }
    nodes.update(updateArray);
    var edgeUpdateArray = [];
    for (let edgeId in allEdges) {
      if (allEdges.hasOwnProperty(edgeId)) {
        edgeUpdateArray.push(allEdges[edgeId]);
      }
    }
    edges.update(edgeUpdateArray);
  } else {
    for (let nodeId in allNodes) {
      if (allNodes.hasOwnProperty(nodeId)) {
        updateArray.push(allNodes[nodeId]);
      }
    }
    nodes.update(updateArray);
    var edgeUpdateArray = [];
    for (let edgeId in allEdges) {
      if (allEdges.hasOwnProperty(edgeId)) {
        edgeUpdateArray.push(allEdges[edgeId]);
      }
    }
    edges.update(edgeUpdateArray);
  }
}

function selectNode(nodes) {
  window._intervalHighlightedIds = [] // Clear any existing highlight intervals when a new node is selected
  network.selectNodes(nodes);
  neighbourhoodHighlight({ nodes: nodes });
  return nodes;
}

function selectNodes(nodes) {
  network.selectNodes(nodes);
  filterHighlight({nodes: nodes});
  return nodes;
}

function selectEdges(edgeIds) {
  filterHighlight({nodes: [], edges: edgeIds});
  return edgeIds;
}

function highlightFilter(filter) {
  var selectedNodes = [];
  var selectedEdges = [];
  var selectedProp = filter['property'];
  var operator = filter['operator'] || 'eq';
  var filterValues = filter['value'];
  var allNodes = nodes.get({ returnType: "Object" });
  var allEdges = edges.get({ returnType: "Object" });
  var isEdgeProp = selectedProp === 'weight' || selectedProp === 'genomic_dist_mb';

  if (isEdgeProp) {
    for (var edgeId in allEdges) {
      var rawEdgeVal = allEdges[edgeId][selectedProp];
      if ((rawEdgeVal === null || rawEdgeVal === undefined) && selectedProp === 'weight') {
        // Backward compatibility with previous serialization.
        rawEdgeVal = allEdges[edgeId].value;
      }
      if (rawEdgeVal === null || rawEdgeVal === undefined) continue;

      var edgeMatched = false;
      if (operator === 'eq') {
        edgeMatched = filterValues.includes(rawEdgeVal.toString());
      } else if (operator === 'gt' || operator === 'lt') {
        if (filterValues.length === 0) continue;
        var edgeThreshold = parseFloat(filterValues[0]);
        var edgeNumVal = parseFloat(rawEdgeVal);
        if (isNaN(edgeThreshold) || isNaN(edgeNumVal)) continue;
        edgeMatched = operator === 'gt' ? edgeNumVal > edgeThreshold : edgeNumVal < edgeThreshold;
      }

      if (edgeMatched) selectedEdges.push(edgeId);
    }
    selectEdges(selectedEdges);
    return;
  }

  for (var nodeId in allNodes) {
    var rawVal = allNodes[nodeId][selectedProp];
    if (rawVal === null || rawVal === undefined) continue;

    var matched = false;
    if (operator === 'eq') {
      // Exact match against any of the selected values (string comparison)
      matched = filterValues.includes(rawVal.toString());
    } else if (operator === 'gt' || operator === 'lt') {
      // Numeric threshold: use the first selected/typed value
      if (filterValues.length === 0) continue;
      var threshold = parseFloat(filterValues[0]);
      var numVal = parseFloat(rawVal);
      if (isNaN(threshold) || isNaN(numVal)) continue;
      matched = operator === 'gt' ? numVal > threshold : numVal < threshold;
    }

    if (matched) selectedNodes.push(nodeId);
  }
  selectNodes(selectedNodes);
}