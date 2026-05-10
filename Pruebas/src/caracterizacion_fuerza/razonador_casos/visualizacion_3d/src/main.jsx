import React, { Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Canvas, useFrame } from '@react-three/fiber';
import { Billboard, Box, Html, Line, OrbitControls, PerspectiveCamera, Stars, Text } from '@react-three/drei';
import { Activity, BrainCircuit, Filter, Focus, Gauge, Pause, Play, Radio, Search, X } from 'lucide-react';
import { create } from 'zustand';
import * as THREE from 'three';
import './styles.css';

const useMemoryStore = create((set) => ({
  selectedId: null,
  hoveredId: null,
  query: '',
  roleFilter: 'todos',
  autoplay: true,
  setSelectedId: (selectedId) => set({ selectedId }),
  setHoveredId: (hoveredId) => set({ hoveredId }),
  setQuery: (query) => set({ query }),
  setRoleFilter: (roleFilter) => set({ roleFilter }),
  setAutoplay: (autoplay) => set({ autoplay }),
}));

const roleColors = {
  caso_de_corte: '#4fffd6',
  caso_con_repeticion_o_duplicado: '#ffd166',
  referencia_excitacion_controlada: '#ff6b9f',
  'Bouc-Wen': '#7aa7ff',
  'Bouc-Wen viscoso': '#6be4ff',
  'Bouc-Wen envolvente': '#00f5a0',
  'KAN-PINN': '#ff6bd6',
  'KAN-PINN envolvente': '#f6ff6b',
  Duhem: '#ffa35c',
  validacion: '#ffffff',
  comparacion: '#c79bff',
  'comparacion envolvente': '#d9ff9a',
  'VLM experto': '#2ecc71',
};

function number(value, digits = 3) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'n/d';
  return value.toFixed(digits);
}

function useMemoryData() {
  const [state, setState] = useState({ data: null, error: null, liveCount: 0 });

  useEffect(() => {
    let cancelled = false;

    const load = () => {
      fetch('/memory_cases.json')
        .then((response) => {
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          return response.json();
        })
        .then((data) => {
          if (!cancelled) {
            setState((prev) => ({
              data,
              error: null,
              liveCount: data.neurona_vlm_count ?? prev.liveCount ?? 0,
            }));
          }
        })
        .catch((error) => {
          if (!cancelled) setState((prev) => ({ ...prev, error }));
        });
    };

    load();
    const interval = setInterval(load, 8000); // refrescar cada 8s
    return () => { cancelled = true; clearInterval(interval); };
  }, []);

  return state;
}

function MemoryCase({ item, selected, related, muted }) {
  const mesh = useRef();
  const setSelectedId = useMemoryStore((state) => state.setSelectedId);
  const setHoveredId = useMemoryStore((state) => state.setHoveredId);
  const hoveredId = useMemoryStore((state) => state.hoveredId);
  const isHovered = hoveredId === item.id;
  const isArtifact = item.metadata.memory_kind === 'artifact';
  const isNeurona = item.metadata.memory_kind === 'neurona_vlm';
  const role = item.metadata.artifact_role || item.metadata.thesis_role;
  const neuronaColor = item.metadata.color || roleColors[role] || '#2ecc71';
  const color = isNeurona ? neuronaColor : (roleColors[role] || '#8ab4ff');
  const entropy = item.entropy.energy_entropy ?? 0;
  const scale = selected ? 1.45 : related ? 1.18 : 1.0;

  useFrame((clock) => {
    if (!mesh.current) return;
    const t = clock.clock.elapsedTime;
    mesh.current.rotation.y = Math.sin(t * 0.55 + item.position[0]) * 0.08;
    mesh.current.position.y = item.position[1] + Math.sin(t * 1.2 + item.position[2]) * 0.07;
    if (isNeurona) {
      const pulse = 1 + Math.sin(t * 3.5 + item.position[0]) * 0.15;
      mesh.current.scale.setScalar(isNeurona ? 0.55 * scale * pulse : scale);
    }
  });

  if (isNeurona) {
    return (
      <group position={item.position}>
        <mesh
          ref={mesh}
          scale={[0.5 * scale, 0.5 * scale, 0.5 * scale]}
          onClick={(event) => {
            event.stopPropagation();
            setSelectedId(item.id);
          }}
          onPointerOver={(event) => {
            event.stopPropagation();
            setHoveredId(item.id);
            document.body.style.cursor = 'pointer';
          }}
          onPointerOut={() => {
            setHoveredId(null);
            document.body.style.cursor = 'default';
          }}
        >
          <sphereGeometry args={[1, 20, 20]} />
          <meshStandardMaterial
            color={color}
            emissive={color}
            emissiveIntensity={selected || isHovered ? 1.6 : 0.9}
            transparent
            opacity={muted ? 0.12 : selected ? 0.95 : 0.75}
            roughness={0.15}
            metalness={0.1}
          />
        </mesh>
        <Billboard position={[0, 0.55, 0]} follow>
          <Text
            fontSize={selected ? 0.09 : 0.07}
            color={muted ? '#4a6a5a' : '#b8ffd8'}
            maxWidth={0.85}
            textAlign="center"
            anchorX="center"
            anchorY="middle"
          >
            {item.metadata.title || item.id}
          </Text>
        </Billboard>
        {(selected || isHovered) && (
          <Html position={[0, -0.55, 0]} center distanceFactor={9}>
            <div className="case-tag neurona-tag">
              neurona VLM · {item.metadata.quality || '?'} · {item.metadata.cutter_condition || '?'}
            </div>
          </Html>
        )}
      </group>
    );
  }

  return (
    <group position={item.position}>
      <mesh
        ref={mesh}
        scale={[isArtifact ? 1.05 * scale : 0.86 * scale, isArtifact ? 0.66 * scale : 0.52 * scale, isArtifact ? 0.055 * scale : 0.038 * scale]}
        onClick={(event) => {
          event.stopPropagation();
          setSelectedId(item.id);
        }}
        onPointerOver={(event) => {
          event.stopPropagation();
          setHoveredId(item.id);
          document.body.style.cursor = 'pointer';
        }}
        onPointerOut={() => {
          setHoveredId(null);
          document.body.style.cursor = 'default';
        }}
      >
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={selected || isHovered ? 1.25 : related ? 0.65 : isArtifact ? 0.45 : 0.25}
          transparent
          opacity={muted ? 0.16 : selected ? 0.94 : isArtifact ? 0.72 : 0.58 + entropy * 0.24}
          roughness={0.25}
          metalness={0.35}
        />
      </mesh>
      <Billboard position={[0, 0, 0.08]} follow>
        <Text
          fontSize={selected ? 0.11 : isArtifact ? 0.078 : 0.086}
          color={muted ? '#6c7892' : '#e8fbff'}
          maxWidth={0.92}
          textAlign="center"
          anchorX="center"
          anchorY="middle"
        >
          {(item.metadata.title || item.id).replace('artifact_', '').replace('corte_', '')}
        </Text>
      </Billboard>
      {(selected || isHovered) && (
        <Html position={[0, -0.65, 0]} center distanceFactor={9}>
          <div className="case-tag">
            {isArtifact ? 'artefacto' : 'similitud max'} {isArtifact ? role : number(item.neighbors[0]?.similarity, 2)}
          </div>
        </Html>
      )}
    </group>
  );
}

function MemoryLinks({ casesById, links, selectedId }) {
  const selected = casesById.get(selectedId);
  const relatedIds = new Set(selected?.neighbors.map((item) => item.case_id) ?? []);

  return links.map((link) => {
    const source = casesById.get(link.source);
    const target = casesById.get(link.target);
    if (!source || !target) return null;
    const active = selectedId && (link.source === selectedId || link.target === selectedId || relatedIds.has(link.source) || relatedIds.has(link.target));
    const opacity = active ? 0.76 : 0.18 + link.similarity * 0.28;
    return (
      <Line
        key={`${link.source}-${link.target}`}
        points={[source.position, target.position]}
        color={active ? '#ffffff' : '#45f5cf'}
        lineWidth={active ? 1.8 : 0.7}
        transparent
        opacity={opacity}
      />
    );
  });
}

function AutoplayRig({ cases }) {
  const autoplay = useMemoryStore((state) => state.autoplay);
  const selectedId = useMemoryStore((state) => state.selectedId);
  const setSelectedId = useMemoryStore((state) => state.setSelectedId);
  const indexRef = useRef(0);
  const timerRef = useRef(0);

  useFrame((state, delta) => {
    if (!autoplay || cases.length === 0) return;
    timerRef.current += delta;
    if (timerRef.current > 3.4) {
      timerRef.current = 0;
      indexRef.current = (indexRef.current + 1) % cases.length;
      setSelectedId(cases[indexRef.current].id);
    }
    const current = cases.find((item) => item.id === selectedId) ?? cases[indexRef.current];
    const target = new THREE.Vector3(...current.position);
    const phase = state.clock.elapsedTime * 0.28;
    const cameraTarget = target.clone().add(new THREE.Vector3(Math.cos(phase) * 5.4, 3.1, Math.sin(phase) * 5.4));
    state.camera.position.lerp(cameraTarget, 0.025);
    state.camera.lookAt(target);
  });

  return null;
}

function MemoryBox() {
  return (
    <group>
      <Box args={[15, 10, 15]}>
        <meshBasicMaterial color="#7fffe7" wireframe transparent opacity={0.13} />
      </Box>
      <gridHelper args={[15, 15, '#2ee8c5', '#173847']} position={[0, -5, 0]} />
    </group>
  );
}

function Scene({ data, visibleCases }) {
  const selectedId = useMemoryStore((state) => state.selectedId);
  const setSelectedId = useMemoryStore((state) => state.setSelectedId);
  const casesById = useMemo(() => new Map(data.cases.map((item) => [item.id, item])), [data.cases]);
  const selected = casesById.get(selectedId);
  const relatedIds = new Set(selected?.neighbors.map((item) => item.case_id) ?? []);
  const visibleIds = new Set(visibleCases.map((item) => item.id));

  return (
    <Canvas shadows dpr={[1, 1.8]} onPointerMissed={() => setSelectedId(null)}>
      <PerspectiveCamera makeDefault position={[9, 6, 10]} fov={48} />
      <color attach="background" args={['#03070d']} />
      <fog attach="fog" args={['#061017', 8, 23]} />
      <ambientLight intensity={0.42} />
      <pointLight position={[5, 7, 5]} intensity={1.9} color="#8fffee" />
      <pointLight position={[-6, -2, -5]} intensity={0.8} color="#ff6b9f" />
      <Suspense fallback={null}>
        <Stars radius={80} depth={42} count={1300} factor={3} saturation={0} fade speed={0.45} />
        <MemoryBox />
        <MemoryLinks casesById={casesById} links={data.links} selectedId={selectedId} />
        {data.cases.map((item) => (
          <MemoryCase
            key={item.id}
            item={item}
            selected={item.id === selectedId}
            related={relatedIds.has(item.id)}
            muted={!visibleIds.has(item.id) || (selectedId && item.id !== selectedId && !relatedIds.has(item.id))}
          />
        ))}
        <AutoplayRig cases={visibleCases} />
      </Suspense>
      <OrbitControls enableDamping dampingFactor={0.07} minDistance={4} maxDistance={22} />
    </Canvas>
  );
}

function Metric({ label, value, icon: Icon = Gauge }) {
  return (
    <div className="metric">
      <Icon size={16} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function DetailsPanel({ data }) {
  const selectedId = useMemoryStore((state) => state.selectedId);
  const setSelectedId = useMemoryStore((state) => state.setSelectedId);
  const selected = data.cases.find((item) => item.id === selectedId) ?? data.cases[0];
  const meta = selected.metadata;
  const isArtifact = meta.memory_kind === 'artifact';
  const isNeurona = meta.memory_kind === 'neurona_vlm';

  return (
    <aside className="panel details">
      <div className="panel-title">
        <div>
          <span className="eyebrow">{isNeurona ? 'Neurona VLM en vivo' : 'Caso activo'}</span>
          <h2>{meta.title || selected.id}</h2>
        </div>
        <button className="icon-button" onClick={() => setSelectedId(null)} title="Limpiar seleccion">
          <X size={18} />
        </button>
      </div>

      {isArtifact && meta.artifact_url && (
        <a className="artifact-preview" href={meta.artifact_url} target="_blank" rel="noreferrer" title="Abrir grafica completa">
          <img src={meta.artifact_url} alt={meta.title || meta.source_file} />
        </a>
      )}

      {isNeurona && (
        <div className="neurona-badge" style={{ borderColor: meta.color || '#2ecc71' }}>
          <span className="neurona-condition">{meta.cutter_condition || '?'}</span>
          <span className="neurona-rpm">{meta.rpm || '?'} rpm</span>
          <span className={`neurona-quality quality-${meta.quality || '?'}`}>{meta.quality || '?'}</span>
        </div>
      )}

      <div className="metric-grid">
        {isNeurona ? (
          <>
            <Metric label="confianza" value={number(meta.confidence, 2)} icon={Activity} />
            <Metric label="area lazo" value={number(meta.area_hysteresis, 3)} icon={Radio} />
            <Metric label="RPM" value={meta.rpm || '?'} icon={Gauge} />
            <Metric label="cortando" value={meta.cutting_state ? 'si' : 'no'} />
          </>
        ) : (
          <>
            <Metric label={isArtifact ? 'alpha' : 'energia H'} value={number(selected.feature_vector.alpha ?? selected.entropy.energy_entropy)} icon={Activity} />
            <Metric label={isArtifact ? 'R2 corte' : 'fuerza H'} value={number(selected.feature_vector.r2_corte ?? selected.entropy.force_entropy)} icon={Radio} />
            <Metric label={isArtifact ? 'R2 shaker' : 'lazo'} value={number(selected.feature_vector.r2_shaker ?? meta.loop_area_norm)} />
            <Metric label="corr" value={number(selected.feature_vector.corr_corte ?? meta.corr_force_input)} />
          </>
        )}
      </div>

      <section>
        <h3>Vecinos equivalentes</h3>
        <div className="neighbor-list">
          {selected.neighbors.map((neighbor) => (
            <button key={neighbor.case_id} className="neighbor" onClick={() => setSelectedId(neighbor.case_id)}>
              <span>{neighbor.case_id}</span>
              <strong>{number(neighbor.similarity, 3)}</strong>
              <small>{neighbor.reasons.join(' · ')}</small>
            </button>
          ))}
        </div>
      </section>

      <section>
        <h3>Metadata</h3>
        <dl className="metadata">
          <dt>fuente</dt>
          <dd>{meta.source || meta.source_file}</dd>
          <dt>tipo</dt>
          <dd>{meta.artifact_role || meta.thesis_role}</dd>
          {isNeurona ? (
            <>
              <dt>modelo VLM</dt>
              <dd>{meta.vlm_model || '?'}</dd>
              <dt>timestamp</dt>
              <dd>{meta.timestamp || '?'}</dd>
              <dt>guardado</dt>
              <dd>{meta.should_store ? 'si (experto aprobo)' : 'no (experto descarto)'}</dd>
              <dt>notas VLM</dt>
              <dd>{meta.notes || 'sin notas'}</dd>
            </>
          ) : isArtifact ? (
            <>
              <dt>archivo</dt>
              <dd>{meta.source_relpath}</dd>
              <dt>ecuacion</dt>
              <dd>{meta.equation || 'n/d'}</dd>
              <dt>lectura</dt>
              <dd>{meta.interpretation || 'artefacto de validacion/modelado'}</dd>
            </>
          ) : (
            <>
              <dt>rpm estimada</dt>
              <dd>{number(meta.rpm_estimada, 2)}</dd>
              <dt>ventana</dt>
              <dd>{number(meta.window_start_s, 3)} - {number(meta.window_end_s, 3)} s</dd>
              <dt>features</dt>
              <dd>{meta.features_relpath}</dd>
            </>
          )}
        </dl>
      </section>

      {meta.suggested_boucwen_command && (
        <section>
          <h3>Bouc-Wen</h3>
          <code className="command">{meta.suggested_boucwen_command}</code>
        </section>
      )}
    </aside>
  );
}

function ControlPanel({ data, visibleCases }) {
  const query = useMemoryStore((state) => state.query);
  const setQuery = useMemoryStore((state) => state.setQuery);
  const roleFilter = useMemoryStore((state) => state.roleFilter);
  const setRoleFilter = useMemoryStore((state) => state.setRoleFilter);
  const autoplay = useMemoryStore((state) => state.autoplay);
  const setAutoplay = useMemoryStore((state) => state.setAutoplay);
  const roles = ['todos', ...Array.from(new Set(data.cases.map((item) => item.metadata.artifact_role || item.metadata.thesis_role)))];

  return (
    <aside className="panel controls">
      <div className="brand">
        <BrainCircuit size={28} />
        <div>
          <h1>Conciencia digital</h1>
          <p>
            {data.experimental_case_count} cortes + {data.artifact_case_count} artefactos
            {data.neurona_vlm_count > 0 && (
              <span className="neurona-count"> + {data.neurona_vlm_count} neuronas VLM</span>
            )}
            {' '}desde {data.generated_from}
          </p>
        </div>
      </div>

      <div className="search">
        <Search size={16} />
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Buscar caso, rpm, fuente" />
      </div>

      <label className="select-label">
        <Filter size={16} />
        <select value={roleFilter} onChange={(event) => setRoleFilter(event.target.value)}>
          {roles.map((role) => (
            <option key={role} value={role}>{role}</option>
          ))}
        </select>
      </label>

      <button className={`autoplay ${autoplay ? 'active' : ''}`} onClick={() => setAutoplay(!autoplay)}>
        {autoplay ? <Pause size={17} /> : <Play size={17} />}
        <span>{autoplay ? 'autoplay pensando' : 'autoplay pausado'}</span>
      </button>

      <div className="status-row">
        <Metric label="visibles" value={`${visibleCases.length}/${data.case_count}`} icon={Focus} />
        <Metric label="enlaces" value={data.links.length} icon={Radio} />
      </div>

      <section>
        <h3>Modelo de equivalencia</h3>
        <p className="model-text">{data.equivalence_model.description}</p>
      </section>
    </aside>
  );
}

function App() {
  const { data, error } = useMemoryData();
  const query = useMemoryStore((state) => state.query).toLowerCase().trim();
  const roleFilter = useMemoryStore((state) => state.roleFilter);

  const visibleCases = useMemo(() => {
    if (!data) return [];
    return data.cases.filter((item) => {
      const role = item.metadata.artifact_role || item.metadata.thesis_role;
      const haystack = `${item.id} ${item.metadata.title || ''} ${item.metadata.source_file} ${item.metadata.rpm_estimada || ''} ${role} ${item.metadata.equation || ''}`.toLowerCase();
      const matchesQuery = !query || haystack.includes(query);
      const matchesRole = roleFilter === 'todos' || role === roleFilter;
      return matchesQuery && matchesRole;
    });
  }, [data, query, roleFilter]);

  if (error) {
    return <main className="loading">No pude cargar <code>memory_cases.json</code>. Ejecuta <code>npm run prepare:data</code>.</main>;
  }

  if (!data) {
    return <main className="loading">Sincronizando recuerdos del razonador...</main>;
  }

  return (
    <main className="app-shell">
      <Scene data={data} visibleCases={visibleCases} />
      <ControlPanel data={data} visibleCases={visibleCases} />
      <DetailsPanel data={data} />
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
