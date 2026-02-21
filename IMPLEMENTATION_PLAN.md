# Implementation Plan

## Document Information
- **Created**: 2026-02-14 14:47:36
- **Purpose**: Detailed implementation roadmap for two-hop relay communication system
- **Reference**: See INPUT.md for complete requirements
- **Status**: In Progress

---

## Project Overview

Building a Python framework for a two-hop digital communication system with classical Amplify and Forward (AF) relay over AWGN channels. This serves as a baseline for future Generative AI relay implementations.

---

## Architecture Design

### System Components

```
┌────────┐    AWGN     ┌───────┐    AWGN     ┌─────────────┐
│ Source │ ──────────> │ Relay │ ──────────> │ Destination │
└────────┘   Channel   └───────┘   Channel   └─────────────┘
    │                      │                        │
    ├─ Generate bits       ├─ Receive signal        ├─ Receive signal
    ├─ Modulate (BPSK)     ├─ Amplify               ├─ Demodulate
    └─ Transmit            └─ Forward               └─ Decode bits
```

### Module Breakdown

#### 1. channel.py
- **Responsibility**: Simulate wireless channel effects
- **Key Function**: `awgn_channel(signal, snr_db)`
- **Theory**: Adds white Gaussian noise with specified SNR
- **Formula**: $\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right)$

#### 2. modulation.py
- **Responsibility**: Convert bits ↔ symbols
- **Key Functions**: 
  - `bpsk_modulate(bits)`: Maps [0,1] → [-1,+1]
  - `bpsk_demodulate(symbols)`: Hard decision based on sign
- **Theory**: BPSK uses antipodal signaling for optimal performance

#### 3. nodes.py
- **Responsibility**: Implement communication entities
- **Classes**:
  - `Source`: Bit generation + modulation
  - `Relay` (base class): Interface for relay strategies
  - `AmplifyAndForwardRelay`: Classical AF implementation
  - `Destination`: Demodulation + bit recovery

#### 4. simulation.py
- **Responsibility**: End-to-end system simulation
- **Key Functions**:
  - Monte Carlo loop over multiple trials
  - BER calculation: $\text{BER} = \frac{\text{bit errors}}{\text{total bits}}$
  - SNR sweep for performance curves

---

## Implementation Checkpoints

### ✅ Checkpoint 00: Documentation Setup
- [x] Create INPUT.md
- [x] Create IMPLEMENTATION_PLAN.md
- [x] Create CHECKPOINT_LOG.md
- [x] Create CHANGELOG.md

### ⏳ Checkpoint 01: AWGN Channel
- [ ] Create `checkpoints/checkpoint_01_channel.py`
- [ ] Implement `awgn_channel(signal, snr_db)` function
- [ ] Implement SNR calculation helpers
- [ ] Add unit test for noise power verification
- [ ] Validate: Measured SNR ≈ Target SNR
- [ ] Update CHECKPOINT_LOG.md with results
- [ ] Update CHANGELOG.md with changes

**Expected Deliverables**:
- Standalone Python file that can be run independently
- Test output showing SNR verification
- Documentation of any assumptions made

### ⏳ Checkpoint 02: BPSK Modulation
- [ ] Create `checkpoints/checkpoint_02_modulation.py`
- [ ] Implement `bpsk_modulate(bits)` function
- [ ] Implement `bpsk_demodulate(symbols)` function
- [ ] Add unit test for modulation/demodulation round-trip
- [ ] Validate: 100% accuracy for noiseless case
- [ ] Update CHECKPOINT_LOG.md with results
- [ ] Update CHANGELOG.md with changes

**Expected Deliverables**:
- Standalone Python file with modulation functions
- Test showing perfect recovery without noise
- Integration notes for next checkpoint

### ⏳ Checkpoint 03: Communication Nodes
- [ ] Create `checkpoints/checkpoint_03_nodes.py`
- [ ] Implement `Source` class
- [ ] Implement `Relay` base class
- [ ] Implement `AmplifyAndForwardRelay` class
- [ ] Implement `Destination` class
- [ ] Add unit tests for each node
- [ ] Validate: Each node performs expected operations
- [ ] Update CHECKPOINT_LOG.md with results
- [ ] Update CHANGELOG.md with changes

**Expected Deliverables**:
- Node classes with clear interfaces
- AF relay with power normalization
- Test showing node interactions

### ⏳ Checkpoint 04: Full Simulation
- [ ] Create `checkpoints/checkpoint_04_simulation.py`
- [ ] Implement Monte Carlo simulation loop
- [ ] Implement BER calculation
- [ ] Implement SNR sweep (e.g., 0 to 20 dB)
- [ ] Integrate all components: Source → Channel → Relay → Channel → Destination
- [ ] Validate: Simulation runs without errors
- [ ] Update CHECKPOINT_LOG.md with results
- [ ] Update CHANGELOG.md with changes

**Expected Deliverables**:
- Complete end-to-end simulation
- BER results for multiple SNR points
- Performance data ready for plotting

### ⏳ Checkpoint 05: Visualization & Validation
- [ ] Create `checkpoints/checkpoint_05_plotting.py`
- [ ] Implement BER vs SNR plotting
- [ ] Compare with theoretical AF relay performance
- [ ] Generate publication-quality plots
- [ ] Validate: BER curve matches expected behavior
- [ ] Update CHECKPOINT_LOG.md with results
- [ ] Update CHANGELOG.md with changes

**Expected Deliverables**:
- BER plot (log scale)
- Comparison with theory
- Analysis of results

---

## Testing Strategy

### Unit Testing
Each checkpoint includes self-contained tests:
```python
if __name__ == "__main__":
    # Test code here
    print("Test passed ✓")
```

### Integration Testing
- Checkpoint 04 tests all components together
- Verify data flows correctly through the system
- Check for numerical stability

### Validation Testing
- Compare BER results with theoretical predictions
- Verify AF relay behavior matches literature
- Check edge cases (very low/high SNR)

---

## Code Quality Standards

### Documentation
- Docstrings for all functions and classes
- Inline comments for complex logic
- Type hints where appropriate

### Naming Conventions
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`
- Private methods: `_leading_underscore`

### Code Structure
```python
# Imports
import numpy as np

# Constants
CONSTANT_NAME = value

# Functions
def function_name(param):
    """Docstring."""
    pass

# Classes
class ClassName:
    """Docstring."""
    pass

# Main execution
if __name__ == "__main__":
    # Test code
    pass
```

---

## Dependencies Management

### Required Packages
```bash
pip install numpy matplotlib scipy
```

### Version Compatibility
- Python: 3.8+
- NumPy: 1.20+
- Matplotlib: 3.3+
- SciPy: 1.6+ (optional)

---

## Rollback Strategy

### If Checkpoint Fails
1. Review CHECKPOINT_LOG.md for error details
2. Check CHANGELOG.md to see what was changed
3. Delete problematic checkpoint file
4. Revert to previous working checkpoint
5. Analyze failure and adjust approach
6. Document lessons learned

### Checkpoint Independence
- Each checkpoint is self-contained
- Can run any checkpoint independently
- No dependencies between checkpoint files
- Final integration happens in Checkpoint 04

---

## Success Metrics

### Per Checkpoint
- ✅ Code executes without errors
- ✅ Unit tests pass
- ✅ Output matches expectations
- ✅ Documentation updated

### Overall Project
- ✅ End-to-end transmission works
- ✅ BER calculation is accurate
- ✅ Performance matches theory
- ✅ Code is modular and extensible
- ✅ Ready for GenAI integration

---

## Timeline Estimate

| Checkpoint | Estimated Time | Complexity |
|------------|---------------|------------|
| 00: Documentation | 30 min | Low |
| 01: AWGN Channel | 45 min | Low |
| 02: BPSK Modulation | 45 min | Low |
| 03: Nodes | 90 min | Medium |
| 04: Simulation | 60 min | Medium |
| 05: Plotting | 30 min | Low |
| **Total** | **5 hours** | - |

*Note: Times are estimates and may vary based on debugging needs*

---

## Future Enhancements

### Phase 2: GenAI Relay
- Replace `AmplifyAndForwardRelay` with `GenAIRelay`
- Train neural network for signal processing
- Compare performance: Classical vs GenAI
- Optimize for various channel conditions

### Additional Features
- Support for QPSK, 16-QAM modulation
- Rayleigh fading channel
- Multiple relay configurations
- Adaptive relay strategies
- Real-time visualization

---

## Notes

### Design Decisions
- **BPSK chosen**: Simplest modulation, well-understood theory
- **AF relay chosen**: Easiest classical relay to implement
- **Checkpoint approach**: Enables traceability and rollback
- **Standalone files**: Each checkpoint can run independently

### Assumptions
- Perfect synchronization (no timing errors)
- Perfect channel state information at relay
- Flat fading (frequency non-selective)
- No inter-symbol interference

---

## Progress Tracking

**Last Updated**: 2026-02-14 14:47:36

**Current Status**: Checkpoint 00 Complete

**Next Action**: Begin Checkpoint 01 (AWGN Channel implementation)

---

**End of Implementation Plan**
