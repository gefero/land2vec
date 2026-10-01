export const meta = {
  name: 'auditoria-land2vec',
  description: 'Equipo de agentes (.claude/agents/) que audita código, estrategia de validación y resultados de land2vec y genera docs/reporte_auditoria.md',
  whenToUse: 'Auditoría completa del proyecto: código embeddings → clusters → viz, validación interna/externa y reporte',
  phases: [
    { title: 'Auditoría', detail: 'auditor-embeddings, auditor-clustering, auditor-viz, evaluador-validacion, analista-resultados' },
    { title: 'Verificación', detail: 'verificador adversarial por área' },
    { title: 'Supervisión', detail: 'supervisor revisa cobertura y coherencia, encarga seguimientos' },
    { title: 'Seguimiento', detail: 'hasta 2 tareas encargadas por el supervisor' },
    { title: 'Reporte', detail: 'redactor-reporte redacta el reporte (lo guarda quien corre el workflow)' },
    { title: 'Revisión final', detail: 'supervisor aprueba o pide correcciones (hasta 2 rondas)' },
  ],
}

// args: { fecha: 'YYYY-MM-DD', python?: ruta al intérprete, scratch?: dir temporal,
//         inlineRoles?: true }
// Con inlineRoles, cada agente lee su propio .claude/agents/<rol>.md al arrancar (útil si esos agentes
// todavía no están registrados en la sesión). Sin inlineRoles, se usan los agentes registrados vía agentType.
const A = args || {}
const ENV = [
  A.python ? `Intérprete de Python a usar: ${A.python}` : '',
  A.scratch ? `Directorio para archivos temporales: ${A.scratch}` : '',
].filter(Boolean).join('\n')

function run(role, task, opts) {
  const o = { ...opts }
  let prompt = `${task}\n\n${ENV}`
  if (A.inlineRoles) prompt = `Sos el agente "${role}". Antes de hacer nada, leé tu definición en /workspaces/land2vec/.claude/agents/${role}.md y actuá según ella. Del frontmatter, el campo \`tools\` lista las ÚNICAS herramientas que podés usar.\n\n${prompt}`
  else o.agentType = role
  return agent(prompt, o)
}

const SEV = ['critica', 'alta', 'media', 'baja', 'info']
const FINDINGS = {
  type: 'object',
  properties: {
    resumen: { type: 'string' },
    hallazgos: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          severidad: { type: 'string', enum: SEV },
          titulo: { type: 'string' },
          ubicacion: { type: 'string' },
          descripcion: { type: 'string' },
          evidencia: { type: 'string' },
          impacto_en_resultados: { type: 'string' },
          sugerencia: { type: 'string' },
        },
        required: ['id', 'severidad', 'titulo', 'ubicacion', 'descripcion', 'evidencia', 'impacto_en_resultados', 'sugerencia'],
      },
    },
    datos_clave: { type: 'string', description: 'Tablas o cifras en markdown (vacío si no aplica)' },
  },
  required: ['resumen', 'hallazgos', 'datos_clave'],
}
const VERDICTS = {
  type: 'object',
  properties: {
    veredictos: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          veredicto: { type: 'string', enum: ['confirmado', 'plausible', 'refutado'] },
          severidad_ajustada: { type: 'string', enum: SEV },
          razon: { type: 'string' },
        },
        required: ['id', 'veredicto', 'severidad_ajustada', 'razon'],
      },
    },
    omisiones: { type: 'string' },
  },
  required: ['veredictos', 'omisiones'],
}
const SUPERVISION = {
  type: 'object',
  properties: {
    evaluacion: { type: 'string' },
    contradicciones: { type: 'string' },
    seguimientos: {
      type: 'array', maxItems: 2,
      items: {
        type: 'object',
        properties: {
          titulo: { type: 'string' },
          agente: { type: 'string', enum: ['auditor-embeddings', 'auditor-clustering', 'auditor-viz', 'evaluador-validacion', 'analista-resultados'] },
          tarea: { type: 'string', description: 'Instrucción autocontenida' },
        },
        required: ['titulo', 'agente', 'tarea'],
      },
    },
    instrucciones_redactor: { type: 'string' },
  },
  required: ['evaluacion', 'contradicciones', 'seguimientos', 'instrucciones_redactor'],
}
const REVIEW = {
  type: 'object',
  properties: { aprobado: { type: 'boolean' }, correcciones: { type: 'string' } },
  required: ['aprobado', 'correcciones'],
}

const AREAS = [
  { role: 'auditor-embeddings', titulo: 'Código: pipeline de embeddings' },
  { role: 'auditor-clustering', titulo: 'Código: clustering y validación' },
  { role: 'auditor-viz', titulo: 'Código: visualizaciones' },
  { role: 'evaluador-validacion', titulo: 'Estrategia de validación' },
  { role: 'analista-resultados', titulo: 'Análisis de los resultados de validación' },
]

const results = await pipeline(
  AREAS,
  a => run(a.role, `Hacé tu trabajo completo sobre tu alcance ("${a.titulo}") en /workspaces/land2vec, con la profundidad de una auditoría previa a la publicación del paper.`,
    { label: a.role, phase: 'Auditoría', schema: FINDINGS }),
  (r, a) => {
    if (!r) return null
    const rel = r.hallazgos.filter(h => h.severidad !== 'info')
    if (!rel.length) return { area: a, audit: r, verif: { veredictos: [], omisiones: '' } }
    return run('verificador', `Verificá los hallazgos de ${a.role} (área "${a.titulo}"):\n${JSON.stringify(rel, null, 1)}`,
      { label: `verificador:${a.role}`, phase: 'Verificación', schema: VERDICTS })
      .then(v => ({ area: a, audit: r, verif: v || { veredictos: [], omisiones: '(el verificador falló)' } }))
  },
)

const ok = results.filter(Boolean)
const dropped = AREAS.filter((a, i) => !results[i]).map(a => a.role)
if (dropped.length) log(`Áreas sin resultado: ${dropped.join(', ')}`)

const material = ok.map(({ area, audit, verif }) => {
  const vmap = Object.fromEntries(verif.veredictos.map(v => [v.id, v]))
  return {
    area: area.titulo, agente: area.role, resumen: audit.resumen, datos_clave: audit.datos_clave,
    hallazgos: audit.hallazgos.map(h => ({ ...h, verificacion: vmap[h.id] || { veredicto: h.severidad === 'info' ? 'no verificado (info)' : 'sin veredicto' } })),
    omisiones_verificador: verif.omisiones,
  }
})

phase('Supervisión')
const sup = await run('supervisor', `Revisión intermedia. Material del equipo${dropped.length ? ` (faltan: ${dropped.join(', ')})` : ''}:\n${JSON.stringify(material, null, 1)}`,
  { label: 'supervisor', phase: 'Supervisión', schema: SUPERVISION })
if (!sup) log('El supervisor falló; sigo sin supervisión intermedia')

const seguimientos = sup ? (await parallel(sup.seguimientos.slice(0, 2).map((s, i) => () =>
  run(s.agente, `Tarea de seguimiento encargada por el supervisor: "${s.titulo}".\n${s.tarea}`,
    { label: `seguimiento:${s.agente}`, phase: 'Seguimiento', schema: FINDINGS })
    .then(r => r && { titulo: s.titulo, agente: s.agente, ...r })))).filter(Boolean) : []

// Los subagentes no pueden escribir archivos de reporte: el redactor devuelve el markdown
// y quien corre el workflow lo guarda en docs/reporte_auditoria.md (campo `markdown` del resultado).
const DRAFT = {
  type: 'object',
  properties: {
    markdown: { type: 'string', description: 'Reporte completo en markdown' },
    resumen: { type: 'string', description: 'Máximo 15 líneas: conclusiones, conteo por severidad, 3 bloqueantes; o correcciones aplicadas/rechazadas' },
  },
  required: ['markdown', 'resumen'],
}

phase('Reporte')
let draft = await run('redactor-reporte', `Redactá el reporte (destino: docs/reporte_auditoria.md), fechado ${A.fecha}. NO escribas archivos: devolvelo completo en el campo markdown. Seguí las instrucciones del supervisor. Los seguimientos no pasaron por el verificador: validá lo crítico.

MATERIAL:
${JSON.stringify(material, null, 1)}

SUPERVISIÓN Y SEGUIMIENTOS:
${JSON.stringify({ supervisor: sup, seguimientos }, null, 1)}`,
  { label: 'redactor-reporte', phase: 'Reporte', schema: DRAFT })

phase('Revisión final')
let rev = null
for (let round = 1; draft && round <= 2; round++) {
  rev = await run('supervisor', `Revisión final del reporte (ronda ${round}). Texto del reporte:\n\n${draft.markdown}\n\nMaterial del equipo para contrastar:\n${JSON.stringify({ material, supervisor: sup, seguimientos }, null, 1)}`,
    { label: `supervisor:revisión-${round}`, phase: 'Revisión final', schema: REVIEW })
  if (!rev || rev.aprobado || round === 2) break
  log(`El supervisor pidió correcciones (ronda ${round})`)
  draft = await run('redactor-reporte', `El supervisor pidió estas correcciones al reporte. Verificá cada una contra las fuentes; aplicá las correctas y explicá en el resumen por qué rechazás las demás. NO escribas archivos: devolvé el reporte completo corregido en markdown.\n\nCORRECCIONES:\n${rev.correcciones}\n\nREPORTE ACTUAL:\n${draft.markdown}`,
    { label: `redactor-reporte:corrección-${round}`, phase: 'Revisión final', schema: DRAFT }) || draft
}

return {
  markdown: draft && draft.markdown,
  resumen: draft && draft.resumen,
  aprobado: !!(rev && rev.aprobado),
  revision: rev,
  supervisor: sup && { evaluacion: sup.evaluacion, seguimientos: sup.seguimientos.map(s => `${s.agente}: ${s.titulo}`) },
  conteos: material.map(m => ({ area: m.area, total: m.hallazgos.length, refutados: m.hallazgos.filter(h => h.verificacion.veredicto === 'refutado').length })),
}
