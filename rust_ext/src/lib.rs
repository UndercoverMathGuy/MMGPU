/// mmgpu_rs — Rust-accelerated Phase 1 graph construction for tensormm.
///
/// Python serialises ParsedDatabase into flat integer arrays (symbol IDs),
/// passes them here, and Rust builds all proof graphs in parallel with rayon.
/// No subprocess overhead; all parallelism is via rayon thread pool.
///
/// Interface (see gpu_verifier.py build_all_proof_graphs_rs):
///   build_graphs(
///       # Symbol table
///       num_symbols: int,
///       # Label table — parallel arrays, one entry per label
///       label_types: &[u8],         # 0=f, 1=e, 2=a/p
///       label_f_typecode: &[i32],   # for $f: symbol id of type_code (-1 otherwise)
///       label_f_variable: &[i32],   # for $f: symbol id of variable (-1 otherwise)
///       label_e_expr_offsets: &[i32],  # for $e: CSR into label_e_expr_data
///       label_e_expr_data: &[i32],     # flat expr token ids for all $e labels
///       label_a_nf: &[i32],            # for $a/$p: count of floating hyps
///       label_a_ne: &[i32],            # for $a/$p: count of essential hyps
///       # Theorems — parallel arrays, one entry per theorem
///       thm_proof_offsets: &[i64],     # CSR into thm_proof_data
///       thm_proof_data: &[i32],        # proof_ints for each theorem (-1=Z save)
///       thm_plabel_offsets: &[i32],    # CSR into thm_plabel_data (for compressed proofs)
///       thm_plabel_data: &[i32],       # label ids for proof label lists
///       thm_expr_offsets: &[i32],      # CSR into thm_expr_data
///       thm_expr_data: &[i32],         # expected conclusion token ids
///   ) -> list[tuple | str]
///
/// Returns one entry per theorem: either a tuple of 10 arrays+scalars
/// representing ProofGraph data, or a str error message.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use rayon::prelude::*;

// Node type constants — must match gpu_verifier.py
const NODE_PUSH_F: i8 = 0;
const NODE_PUSH_E: i8 = 1;
const NODE_ASSERTION: i8 = 2;

#[derive(Debug)]
struct GraphResult {
    /// max topological level among assertion nodes
    max_level: i32,
    /// max expression length among push nodes
    max_push_expr_len: i32,
    /// number of nodes
    num_nodes: i32,
    /// per-node type [N] i8
    node_types: Vec<i8>,
    /// per-node level [N] i32
    node_levels: Vec<i32>,
    /// per-node label id [N] i32 (0 for push nodes)
    node_label_ids: Vec<i32>,
    /// CSR offsets [N+1] i32
    input_offsets: Vec<i32>,
    /// CSR data [total_inputs] i32
    input_data: Vec<i32>,
    /// step indices of push nodes [num_push] i32
    push_node_indices: Vec<i32>,
    /// expr token ids for push nodes, flat CSR data [total_push_tokens] i32
    push_expr_data: Vec<i32>,
    /// CSR offsets for push exprs [num_push+1] i32
    push_expr_offsets: Vec<i32>,
    /// label_id_map: pairs of (label_id_in_label_table, local_lid) for assertion nodes
    label_id_map_keys: Vec<i32>,   // global label ids (index into label_types etc.)
    label_id_map_vals: Vec<i32>,   // local compact ids (0-based)
}

fn build_one_graph(
    // label table slices
    label_types: &[u8],
    label_f_typecode: &[i32],
    label_f_variable: &[i32],
    label_e_expr_offsets: &[i32],
    label_e_expr_data: &[i32],
    label_a_nf: &[i32],
    label_a_ne: &[i32],
    // this theorem's proof
    proof_ints: &[i32],
    plabels: &[i32],
    thm_expr: &[i32],
) -> Result<GraphResult, String> {
    let mut node_types: Vec<i8> = Vec::new();
    let mut node_levels: Vec<i32> = Vec::new();
    let mut node_label_ids: Vec<i32> = Vec::new();
    let mut input_offsets: Vec<i32> = vec![0i32];
    let mut input_data: Vec<i32> = Vec::new();
    let mut push_node_indices: Vec<i32> = Vec::new();
    let mut push_expr_data: Vec<i32> = Vec::new();
    let mut push_expr_offsets: Vec<i32> = vec![0i32];

    // label_id_map: global_label_id → local compact id
    let mut label_id_map: std::collections::HashMap<i32, i32> =
        std::collections::HashMap::new();

    let mut virtual_stack: Vec<i32> = Vec::new();
    let mut step_levels: Vec<i32> = Vec::new();
    let mut step_counter: i32 = 0;
    let mut max_level: i32 = 0;
    let mut max_push_expr_len: i32 = 0;
    let mut saved_indices: Vec<i32> = Vec::new();

    // Process one label (given by its global id in the label table)
    macro_rules! process_label {
        ($lid:expr) => {{
            let lid = $lid as usize;
            let ltype = label_types[lid];
            match ltype {
                0 => {
                    // $f: push floating hyp — expr is [type_code, variable]
                    let tc = label_f_typecode[lid];
                    let vr = label_f_variable[lid];
                    let elen = 2i32;
                    if elen > max_push_expr_len {
                        max_push_expr_len = elen;
                    }
                    node_types.push(NODE_PUSH_F);
                    node_levels.push(0);
                    node_label_ids.push(0);
                    input_offsets.push(*input_offsets.last().unwrap());
                    push_node_indices.push(step_counter);
                    push_expr_data.push(tc);
                    push_expr_data.push(vr);
                    push_expr_offsets.push(*push_expr_offsets.last().unwrap() + elen);
                    step_levels.push(0);
                    virtual_stack.push(step_counter);
                    step_counter += 1;
                    None::<String>
                }
                1 => {
                    // $e: push essential hyp
                    let es = label_e_expr_offsets[lid] as usize;
                    let ee = label_e_expr_offsets[lid + 1] as usize;
                    let elen = (ee - es) as i32;
                    if elen > max_push_expr_len {
                        max_push_expr_len = elen;
                    }
                    node_types.push(NODE_PUSH_E);
                    node_levels.push(0);
                    node_label_ids.push(0);
                    input_offsets.push(*input_offsets.last().unwrap());
                    push_node_indices.push(step_counter);
                    push_expr_data.extend_from_slice(&label_e_expr_data[es..ee]);
                    push_expr_offsets.push(*push_expr_offsets.last().unwrap() + elen);
                    step_levels.push(0);
                    virtual_stack.push(step_counter);
                    step_counter += 1;
                    None::<String>
                }
                2 => {
                    // $a or $p: assertion
                    let n_f = label_a_nf[lid] as usize;
                    let n_e = label_a_ne[lid] as usize;
                    let npop = n_f + n_e;
                    if virtual_stack.len() < npop {
                        return Err(format!(
                            "Stack underflow at label {}: need {}, have {}",
                            lid, npop, virtual_stack.len()
                        ));
                    }
                    let stack_len = virtual_stack.len();
                    let input_steps = &virtual_stack[stack_len - npop..];

                    let mut level = 0i32;
                    for &si in input_steps {
                        let lv1 = step_levels[si as usize] + 1;
                        if lv1 > level {
                            level = lv1;
                        }
                    }
                    if level > max_level {
                        max_level = level;
                    }

                    // Assign compact local label id
                    let next_id = label_id_map.len() as i32;
                    let local_lid = *label_id_map
                        .entry(lid as i32)
                        .or_insert(next_id);

                    node_types.push(NODE_ASSERTION);
                    node_levels.push(level);
                    node_label_ids.push(local_lid);
                    for &si in input_steps {
                        input_data.push(si);
                    }
                    input_offsets
                        .push(*input_offsets.last().unwrap() + npop as i32);
                    step_levels.push(level);

                    virtual_stack.truncate(stack_len - npop);
                    virtual_stack.push(step_counter);
                    step_counter += 1;
                    None::<String>
                }
                _ => Some(format!("Unknown label type {} for label {}", ltype, lid)),
            }
        }};
    }

    let label_end = plabels.len();

    for &proof_int in proof_ints {
        if proof_int == -1 {
            // Z save
            if virtual_stack.is_empty() {
                return Err("Z save on empty stack".to_string());
            }
            saved_indices.push(*virtual_stack.last().unwrap());
        } else if (proof_int as usize) < label_end {
            let global_lid = plabels[proof_int as usize];
            if let Some(e) = process_label!(global_lid) {
                return Err(e);
            }
        } else {
            // Reference to saved index
            let si = (proof_int as usize) - label_end;
            if si >= saved_indices.len() {
                return Err(format!(
                    "Saved index {} out of range (only {} saved)",
                    si,
                    saved_indices.len()
                ));
            }
            virtual_stack.push(saved_indices[si]);
        }
    }

    if virtual_stack.len() != 1 {
        return Err(format!(
            "Stack has {} entries at end of proof, expected 1",
            virtual_stack.len()
        ));
    }

    let num_nodes = step_counter;
    let mut lm_keys = Vec::with_capacity(label_id_map.len());
    let mut lm_vals = Vec::with_capacity(label_id_map.len());
    for (k, v) in &label_id_map {
        lm_keys.push(*k);
        lm_vals.push(*v);
    }

    // Ignore thm_expr (used in Python to set expected_conclusion — passed back as-is)
    let _ = thm_expr;

    Ok(GraphResult {
        max_level,
        max_push_expr_len,
        num_nodes,
        node_types,
        node_levels,
        node_label_ids,
        input_offsets,
        input_data,
        push_node_indices,
        push_expr_data,
        push_expr_offsets,
        label_id_map_keys: lm_keys,
        label_id_map_vals: lm_vals,
    })
}

/// Build proof graphs for all theorems in parallel.
///
/// All array arguments are flat Python bytes objects (little-endian).
/// Returns a Python list, one entry per theorem:
///   - On success: a tuple (max_level, max_push_expr_len, num_nodes,
///                          node_types_bytes, node_levels_bytes,
///                          node_label_ids_bytes, input_offsets_bytes,
///                          input_data_bytes, push_node_indices_bytes,
///                          push_expr_data_bytes, push_expr_offsets_bytes,
///                          label_id_map_keys_bytes, label_id_map_vals_bytes)
///   - On error: a str error message
#[pyfunction]
fn build_graphs<'py>(
    py: Python<'py>,
    // label table
    label_types_b: &[u8],
    label_f_typecode_b: &[u8],
    label_f_variable_b: &[u8],
    label_e_expr_offsets_b: &[u8],
    label_e_expr_data_b: &[u8],
    label_a_nf_b: &[u8],
    label_a_ne_b: &[u8],
    // theorems
    num_theorems: usize,
    thm_proof_offsets_b: &[u8],
    thm_proof_data_b: &[u8],
    thm_plabel_offsets_b: &[u8],
    thm_plabel_data_b: &[u8],
    thm_expr_offsets_b: &[u8],
    thm_expr_data_b: &[u8],
) -> PyResult<Bound<'py, PyList>> {
    // Reinterpret byte slices as typed slices
    let label_types = label_types_b;
    let label_f_typecode = bytemuck_cast_slice::<i32>(label_f_typecode_b);
    let label_f_variable = bytemuck_cast_slice::<i32>(label_f_variable_b);
    let label_e_expr_offsets = bytemuck_cast_slice::<i32>(label_e_expr_offsets_b);
    let label_e_expr_data = bytemuck_cast_slice::<i32>(label_e_expr_data_b);
    let label_a_nf = bytemuck_cast_slice::<i32>(label_a_nf_b);
    let label_a_ne = bytemuck_cast_slice::<i32>(label_a_ne_b);

    let thm_proof_offsets = bytemuck_cast_slice::<i64>(thm_proof_offsets_b);
    let thm_proof_data = bytemuck_cast_slice::<i32>(thm_proof_data_b);
    let thm_plabel_offsets = bytemuck_cast_slice::<i32>(thm_plabel_offsets_b);
    let thm_plabel_data = bytemuck_cast_slice::<i32>(thm_plabel_data_b);
    let thm_expr_offsets = bytemuck_cast_slice::<i32>(thm_expr_offsets_b);
    let thm_expr_data = bytemuck_cast_slice::<i32>(thm_expr_data_b);

    // Build all graphs in parallel with rayon
    let results: Vec<Result<GraphResult, String>> = (0..num_theorems)
        .into_par_iter()
        .map(|ti| {
            let ps = thm_proof_offsets[ti] as usize;
            let pe = thm_proof_offsets[ti + 1] as usize;
            let proof_ints = &thm_proof_data[ps..pe];

            let pls = thm_plabel_offsets[ti] as usize;
            let ple = thm_plabel_offsets[ti + 1] as usize;
            let plabels = &thm_plabel_data[pls..ple];

            let es = thm_expr_offsets[ti] as usize;
            let ee = thm_expr_offsets[ti + 1] as usize;
            let thm_expr = &thm_expr_data[es..ee];

            build_one_graph(
                label_types,
                label_f_typecode,
                label_f_variable,
                label_e_expr_offsets,
                label_e_expr_data,
                label_a_nf,
                label_a_ne,
                proof_ints,
                plabels,
                thm_expr,
            )
        })
        .collect();

    // Pack results back into Python objects
    let out = PyList::empty(py);
    for res in results {
        match res {
            Err(e) => out.append(e.into_pyobject(py)?)?,
            Ok(g) => {
                let tup = PyTuple::new(
                    py,
                    &[
                        g.max_level.into_pyobject(py)?.into_any(),
                        g.max_push_expr_len.into_pyobject(py)?.into_any(),
                        g.num_nodes.into_pyobject(py)?.into_any(),
                        bytes_of_i8(&g.node_types).into_pyobject(py)?.into_any(),
                        bytes_of_i32(&g.node_levels).into_pyobject(py)?.into_any(),
                        bytes_of_i32(&g.node_label_ids).into_pyobject(py)?.into_any(),
                        bytes_of_i32(&g.input_offsets).into_pyobject(py)?.into_any(),
                        bytes_of_i32(&g.input_data).into_pyobject(py)?.into_any(),
                        bytes_of_i32(&g.push_node_indices).into_pyobject(py)?.into_any(),
                        bytes_of_i32(&g.push_expr_data).into_pyobject(py)?.into_any(),
                        bytes_of_i32(&g.push_expr_offsets).into_pyobject(py)?.into_any(),
                        bytes_of_i32(&g.label_id_map_keys).into_pyobject(py)?.into_any(),
                        bytes_of_i32(&g.label_id_map_vals).into_pyobject(py)?.into_any(),
                    ],
                )?;
                out.append(tup)?;
            }
        }
    }
    Ok(out)
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn bytemuck_cast_slice<T: bytemuck::Pod>(b: &[u8]) -> &[T] {
    let (prefix, body, suffix) = unsafe { b.align_to::<T>() };
    assert!(prefix.is_empty() && suffix.is_empty(), "unaligned bytes");
    body
}

fn bytes_of_i8(v: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}

fn bytes_of_i32(v: &[i32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
    }
}

// ── $d post-check ─────────────────────────────────────────────────────────────

/// Collect variable sym_ids that appear in a fully-substituted expression.
/// `expr` is a sym_id slice (no type code).
/// For each token: if it's a variable AND in subst, recurse into subst value;
/// otherwise if it's a variable, emit it directly.
fn vars_in_substituted(
    expr: &[i32],
    subst: &std::collections::HashMap<i32, Vec<i32>>,  // var_id → expr (no type code)
    is_variable: &[u8],
) -> Vec<i32> {
    let mut out = Vec::new();
    for &tok in expr {
        let is_var = (tok as usize) < is_variable.len() && is_variable[tok as usize] != 0;
        if is_var {
            if let Some(s) = subst.get(&tok) {
                // substituted — recurse into substitution value
                for &v in s {
                    let sv = (v as usize) < is_variable.len() && is_variable[v as usize] != 0;
                    if sv {
                        out.push(v);
                    }
                }
            } else {
                out.push(tok);
            }
        }
    }
    out
}

/// Apply substitution to an expression (with type code at position 0).
/// Returns the new expression (with type code).
fn apply_subst(
    expr: &[i32],
    subst: &std::collections::HashMap<i32, Vec<i32>>,
) -> Vec<i32> {
    let mut out = Vec::with_capacity(expr.len() * 2);
    for (i, &tok) in expr.iter().enumerate() {
        if i == 0 {
            // type code — always literal
            out.push(tok);
        } else if let Some(s) = subst.get(&tok) {
            out.extend_from_slice(s);
        } else {
            out.push(tok);
        }
    }
    out
}

fn check_dv_one(
    label_types: &[u8],
    label_f_typecode: &[i32],
    label_f_variable: &[i32],
    label_e_expr_offsets: &[i32],
    label_e_expr_data: &[i32],
    label_a_ne: &[i32],
    label_a_fhyp_var_offsets: &[i32],
    label_a_fhyp_var_data: &[i32],
    label_a_expr_offsets: &[i32],
    label_a_expr_data: &[i32],
    label_a_dv_offsets: &[i32],
    label_a_dv_data: &[i32],  // interleaved: x0 y0 x1 y1 ...
    proof_ints: &[i32],
    plabels: &[i32],
    active_dv: &std::collections::HashSet<(i32, i32)>,
    is_variable: &[u8],
) -> Option<String> {
    let mut stack: Vec<Vec<i32>> = Vec::new();
    let mut saved: Vec<Vec<i32>> = Vec::new();
    let label_end = plabels.len();

    for &pi in proof_ints {
        if pi == -1 {
            if stack.is_empty() {
                return Some("Z-save on empty stack".to_string());
            }
            saved.push(stack.last().unwrap().clone());
            continue;
        }

        let pi_u = pi as usize;
        if pi_u >= label_end {
            // backref
            let si = pi_u - label_end;
            if si >= saved.len() {
                return Some(format!("backref {} out of range ({} saved)", si, saved.len()));
            }
            stack.push(saved[si].clone());
            continue;
        }

        let global_lid = plabels[pi_u] as usize;
        match label_types[global_lid] {
            0 => {
                // $f
                stack.push(vec![label_f_typecode[global_lid], label_f_variable[global_lid]]);
            }
            1 => {
                // $e
                let es = label_e_expr_offsets[global_lid] as usize;
                let ee = label_e_expr_offsets[global_lid + 1] as usize;
                stack.push(label_e_expr_data[es..ee].to_vec());
            }
            2 => {
                // $a or $p
                let fv_start = label_a_fhyp_var_offsets[global_lid] as usize;
                let fv_end   = label_a_fhyp_var_offsets[global_lid + 1] as usize;
                let nf = fv_end - fv_start;
                let ne = label_a_ne[global_lid] as usize;
                let npop = nf + ne;

                if stack.len() < npop {
                    return Some(format!(
                        "stack underflow: need {}, have {}", npop, stack.len()
                    ));
                }

                let sp = stack.len() - npop;

                // Build substitution: var_sym_id → expr without type code
                let mut subst: std::collections::HashMap<i32, Vec<i32>> =
                    std::collections::HashMap::with_capacity(nf);
                for f in 0..nf {
                    let var_id = label_a_fhyp_var_data[fv_start + f];
                    let expr_with_tc = &stack[sp + f];
                    // strip type code (index 0)
                    let body = if expr_with_tc.len() > 1 {
                        expr_with_tc[1..].to_vec()
                    } else {
                        Vec::new()
                    };
                    subst.insert(var_id, body);
                }

                // Check $d constraints
                let dv_start = label_a_dv_offsets[global_lid] as usize;
                let dv_end   = label_a_dv_offsets[global_lid + 1] as usize;
                // dv_data is interleaved pairs: x0 y0 x1 y1 ...
                let mut dv_i = dv_start;
                while dv_i + 1 < dv_end {
                    let x = label_a_dv_data[dv_i];
                    let y = label_a_dv_data[dv_i + 1];
                    dv_i += 2;

                    // vars in subst(x) and subst(y)
                    let x_expr: &[i32] = subst.get(&x).map(|v| v.as_slice()).unwrap_or(&[]);
                    let y_expr: &[i32] = subst.get(&y).map(|v| v.as_slice()).unwrap_or(&[]);

                    let sx = if x_expr.is_empty() {
                        // x is unsubstituted — treat as bare variable
                        let is_var = (x as usize) < is_variable.len()
                            && is_variable[x as usize] != 0;
                        if is_var { vec![x] } else { vec![] }
                    } else {
                        vars_in_substituted(x_expr, &subst, is_variable)
                    };

                    let sy = if y_expr.is_empty() {
                        let is_var = (y as usize) < is_variable.len()
                            && is_variable[y as usize] != 0;
                        if is_var { vec![y] } else { vec![] }
                    } else {
                        vars_in_substituted(y_expr, &subst, is_variable)
                    };

                    for &v in &sx {
                        for &w in &sy {
                            if v == w {
                                return Some(format!(
                                    "$d violation: variables ${} and ${} are the same",
                                    x, y
                                ));
                            }
                            let pair = (v.min(w), v.max(w));
                            if !active_dv.contains(&pair) {
                                return Some(format!(
                                    "$d violation: sym {} and sym {} not disjoint",
                                    v, w
                                ));
                            }
                        }
                    }
                }

                // Apply subst to conclusion, push result
                let ae_start = label_a_expr_offsets[global_lid] as usize;
                let ae_end   = label_a_expr_offsets[global_lid + 1] as usize;
                let conclusion = &label_a_expr_data[ae_start..ae_end];
                let result = apply_subst(conclusion, &subst);

                stack.truncate(sp);
                stack.push(result);
            }
            _ => {
                return Some(format!("unknown label type {} for lid {}", label_types[global_lid], global_lid));
            }
        }
    }

    None
}

/// Check $d constraints for all theorems in parallel (rayon).
///
/// Returns a Python list of one entry per theorem:
///   - None  → passed
///   - str   → failure reason
#[pyfunction]
fn check_dv_all<'py>(
    py: Python<'py>,
    // label table (same encoding as build_graphs)
    label_types_b: &[u8],
    label_f_typecode_b: &[u8],
    label_f_variable_b: &[u8],
    label_e_expr_offsets_b: &[u8],
    label_e_expr_data_b: &[u8],
    label_a_ne_b: &[u8],
    // new: fhyp var sym_ids per assertion, CSR
    label_a_fhyp_var_offsets_b: &[u8],
    label_a_fhyp_var_data_b: &[u8],
    // new: conclusion expr per assertion, CSR
    label_a_expr_offsets_b: &[u8],
    label_a_expr_data_b: &[u8],
    // new: mandatory DV pairs per assertion, CSR (interleaved x y)
    label_a_dv_offsets_b: &[u8],
    label_a_dv_data_b: &[u8],
    // theorems
    num_theorems: usize,
    thm_proof_offsets_b: &[u8],
    thm_proof_data_b: &[u8],
    thm_plabel_offsets_b: &[u8],
    thm_plabel_data_b: &[u8],
    // new: active DV pairs per theorem, CSR (canonical min/max pairs, interleaved)
    thm_active_dv_offsets_b: &[u8],
    thm_active_dv_data_b: &[u8],
    // new: is_variable per symbol (u8, 0/1)
    is_variable_b: &[u8],
) -> PyResult<Bound<'py, PyList>> {
    let label_types             = label_types_b;
    let label_f_typecode        = bytemuck_cast_slice::<i32>(label_f_typecode_b);
    let label_f_variable        = bytemuck_cast_slice::<i32>(label_f_variable_b);
    let label_e_expr_offsets    = bytemuck_cast_slice::<i32>(label_e_expr_offsets_b);
    let label_e_expr_data       = bytemuck_cast_slice::<i32>(label_e_expr_data_b);
    let label_a_ne              = bytemuck_cast_slice::<i32>(label_a_ne_b);
    let label_a_fhyp_var_off    = bytemuck_cast_slice::<i32>(label_a_fhyp_var_offsets_b);
    let label_a_fhyp_var_data   = bytemuck_cast_slice::<i32>(label_a_fhyp_var_data_b);
    let label_a_expr_off        = bytemuck_cast_slice::<i32>(label_a_expr_offsets_b);
    let label_a_expr_data       = bytemuck_cast_slice::<i32>(label_a_expr_data_b);
    let label_a_dv_off          = bytemuck_cast_slice::<i32>(label_a_dv_offsets_b);
    let label_a_dv_data         = bytemuck_cast_slice::<i32>(label_a_dv_data_b);

    let thm_proof_offsets       = bytemuck_cast_slice::<i64>(thm_proof_offsets_b);
    let thm_proof_data          = bytemuck_cast_slice::<i32>(thm_proof_data_b);
    let thm_plabel_offsets      = bytemuck_cast_slice::<i32>(thm_plabel_offsets_b);
    let thm_plabel_data         = bytemuck_cast_slice::<i32>(thm_plabel_data_b);
    let thm_active_dv_offsets   = bytemuck_cast_slice::<i32>(thm_active_dv_offsets_b);
    let thm_active_dv_data      = bytemuck_cast_slice::<i32>(thm_active_dv_data_b);
    let is_variable             = is_variable_b;

    // Run all theorems in parallel
    let results: Vec<Option<String>> = (0..num_theorems)
        .into_par_iter()
        .map(|ti| {
            let ps = thm_proof_offsets[ti] as usize;
            let pe = thm_proof_offsets[ti + 1] as usize;
            let proof_ints = &thm_proof_data[ps..pe];

            let pls = thm_plabel_offsets[ti] as usize;
            let ple = thm_plabel_offsets[ti + 1] as usize;
            let plabels = &thm_plabel_data[pls..ple];

            // Build active_dv set for this theorem
            let dv_s = thm_active_dv_offsets[ti] as usize;
            let dv_e = thm_active_dv_offsets[ti + 1] as usize;
            let dv_slice = &thm_active_dv_data[dv_s..dv_e];
            let mut active_dv = std::collections::HashSet::with_capacity((dv_e - dv_s) / 2);
            let mut i = 0;
            while i + 1 < dv_slice.len() {
                // Canonicalize by integer ordering — the serialised data uses
                // string-canonical ordering which may differ from sym_id ordering.
                let a = dv_slice[i];
                let b = dv_slice[i + 1];
                active_dv.insert((a.min(b), a.max(b)));
                i += 2;
            }

            check_dv_one(
                label_types,
                label_f_typecode,
                label_f_variable,
                label_e_expr_offsets,
                label_e_expr_data,
                label_a_ne,
                label_a_fhyp_var_off,
                label_a_fhyp_var_data,
                label_a_expr_off,
                label_a_expr_data,
                label_a_dv_off,
                label_a_dv_data,
                proof_ints,
                plabels,
                &active_dv,
                is_variable,
            )
        })
        .collect();

    // Pack into Python list
    let out = PyList::empty(py);
    for res in results {
        match res {
            None    => out.append(py.None())?,
            Some(s) => out.append(s.into_pyobject(py)?)?,
        }
    }
    Ok(out)
}

#[pymodule]
fn mmgpu_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_graphs, m)?)?;
    m.add_function(wrap_pyfunction!(check_dv_all, m)?)?;
    Ok(())
}
