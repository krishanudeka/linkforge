import React, { useMemo, useRef, useEffect, useState } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'

const TYPE_COLORS = {
  Chemical: '#38bdf8', Protein: '#a78bfa', Gene: '#c084fc', Disease: '#f87171', Process: '#34d399',
  Food: '#fbbf24', Organism: '#86efac', Other: '#94a3b8',
}

export default function GraphCanvas({ nodes, links, seed, onNodeClick, onLinkClick, highlight }) {
  const fg = useRef()
  const box = useRef()
  const [size, setSize] = useState({ w: 600, h: 500 })

  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }))
    if (box.current) ro.observe(box.current)
    return () => ro.disconnect()
  }, [])

  // force-graph mutates its input, so hand it fresh copies
  const data = useMemo(() => ({
    nodes: nodes.map((n) => ({ ...n })),
    links: links.map((l) => ({ ...l })),
  }), [nodes, links])

  useEffect(() => { fg.current?.zoomToFit?.(600, 60) }, [data])

  const linkObject = (l) => {
    const geom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()])
    const mat = l.predicted
      ? new THREE.LineDashedMaterial({ color: 0xf0abfc, dashSize: 3, gapSize: 2, transparent: true, opacity: 0.9 })
      : new THREE.LineBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.35 + 0.5 * Math.min(1, l.confidence || 0.5) })
    return new THREE.Line(geom, mat)
  }
  const updateLink = (line, { start, end }) => {
    line.geometry.setFromPoints([new THREE.Vector3(start.x, start.y, start.z), new THREE.Vector3(end.x, end.y, end.z)])
    line.computeLineDistances()
    return true
  }

  return (
    <div ref={box} className="w-full h-full">
      <ForceGraph3D
        ref={fg}
        width={size.w}
        height={size.h}
        graphData={data}
        backgroundColor="#020617"
        nodeLabel={(n) => `${n.id} (${n.label})`}
        nodeColor={(n) => (highlight?.has(n.id) ? '#ffffff' : TYPE_COLORS[n.label] || TYPE_COLORS.Other)}
        nodeVal={(n) => (n.id === seed ? 14 : 3 + Math.min(8, (n.degree || 1)))}
        nodeOpacity={0.95}
        linkThreeObject={linkObject}
        linkPositionUpdate={updateLink}
        linkLabel={(l) => `${l.type}${l.predicted ? ' (predicted ' + Math.round((l.confidence || 0) * 100) + '%)' : ''}`}
        linkDirectionalParticles={(l) => (l.predicted ? 4 : 0)}
        linkDirectionalParticleColor={() => '#f0abfc'}
        linkDirectionalParticleWidth={1.6}
        onEngineStop={() => fg.current?.zoomToFit?.(400, 60)}
        onNodeClick={(n) => onNodeClick?.(n)}
        onLinkClick={(l) => onLinkClick?.(l)}
      />
    </div>
  )
}
