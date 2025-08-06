# Chatterbox Performance Optimization - Reality Check & Plan

## Current Situation Assessment

### What We Know for Certain
- **Current Performance (Mac CPU)**: RTF 2.728 (very slow, non-real-time)
- **Target Performance (RTX 4090)**: RTF < 1.0 for real-time streaming
- **Confirmed Code Issues**:
  1. `output_attentions=True` hardcoded at line 353 in `src/chatterbox/tts.py`
  2. AlignmentStreamAnalyzer forces attention output, causing "falling back to manual attention" warning
  3. Deprecated KV cache tuple format warnings
  4. No torch.compile() usage anywhere in codebase

### What We Don't Know
- **Actual RTX 4090 performance** without optimizations
- **Real impact** of each optimization (previous docs may overestimate)
- **Whether Flash Attention is actually available** on the target system
- **Actual bottlenecks** in CUDA environment vs CPU testing

## Conservative Optimization Plan

### Phase 1: Minimal Risk Changes (Day 1)
**Goal**: Fix obvious inefficiencies without breaking functionality

1. **Remove Forced Attention Output** (Highest confidence fix)
   ```python
   # In src/chatterbox/tts.py line 353
   output_attentions=False,  # Was: True
   ```
   - **Risk**: Very low
   - **Expected Impact**: Unknown, but removes "falling back" warning
   - **Test Method**: Compare RTF before/after on RTX 4090

2. **Add Performance Mode Flag**
   ```python
   def generate_stream(self, ..., optimize_performance=True):
       if optimize_performance:
           alignment_stream_analyzer = None
       else:
           # Keep existing debugging functionality
   ```
   - **Risk**: Low (maintains backward compatibility)
   - **Expected Impact**: May enable Flash Attention

### Phase 2: Incremental Improvements (Week 1)
**Goal**: Add proven optimizations with measurement

3. **Torch Compilation** (If Phase 1 shows promise)
   ```python
   if hasattr(torch, 'compile') and device == 'cuda':
       self.t3.tfmr = torch.compile(self.t3.tfmr, mode='default')  # Start conservative
   ```
   - **Risk**: Medium (compilation can fail/be slower)
   - **Test Method**: A/B test with/without compilation

4. **KV Cache Modernization**
   ```python
   from transformers import DynamicCache
   # Replace tuple format usage
   ```
   - **Risk**: Medium (API changes)
   - **Expected Impact**: Remove warnings, possible minor speedup

### Phase 3: Data-Driven Decisions (Week 2+)
**Goal**: Only proceed with changes that show measurable improvement

5. **Mixed Precision/Float16** (Only if needed)
6. **Streaming Architecture Changes** (Only if other gains insufficient)

## Measurement Strategy

### Before Any Changes
1. **Baseline RTX 4090 Performance**: Run `performance_test_harness.py` on actual target system
2. **Profile Current Bottlenecks**: Use PyTorch profiler to identify actual slow components

### After Each Change  
1. **A/B Test**: Compare performance with/without each optimization
2. **Monitor for Regressions**: Ensure audio quality maintained
3. **Document Actual Results**: Replace speculation with measurements

## Reality Check Questions

Before implementing, verify:
1. **Is Flash Attention actually available** on your RTX 4090 + PyTorch version?
2. **What's the actual baseline RTF** on RTX 4090 without changes?
3. **Are there other bottlenecks** not visible in CPU testing?

## Success Criteria

- **Minimum**: RTF < 1.0 (real-time capable)
- **Target**: RTF < 0.7 (comfortable real-time margin)
- **Maintain**: Audio quality equivalent to current implementation
- **No Regressions**: Existing functionality preserved

## Risk Mitigation

1. **Version Control**: All changes in feature branches
2. **Rollback Plan**: Keep original implementation available
3. **Incremental Testing**: One change at a time
4. **Documentation**: Record actual results vs predictions

## Next Actions

1. **Get RTX 4090 baseline measurement**
2. **Implement Phase 1 changes only**
3. **Measure real impact before proceeding**
4. **Abandon optimizations that don't deliver measurable improvements**

---

*This plan prioritizes proven, low-risk changes over speculative optimizations. Each phase requires measurement before proceeding to the next.*