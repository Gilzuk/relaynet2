# Project Input Specification

## Document Information
- **Created**: 2026-02-14 14:46:42
- **Purpose**: Master reference document for two-hop relay communication system
- **Status**: Active

---

## 1. Project Goal

Explore the use of Generative AI for relay in digital communication by first building a Python framework for transmit and receive over a simple channel with two hops (source → relay → destination).

### Primary Objective
Build a baseline classical relay system that can later be enhanced with Generative AI techniques.

---

## 2. System Architecture

### Communication Path
```
Source → AWGN Channel → Relay → AWGN Channel → Destination
```

### Two-Hop Configuration
- **Hop 1**: Source to Relay
- **Hop 2**: Relay to Destination

---

## 3. Technical Specifications

### 3.1 Channel Model
- **Type**: AWGN (Additive White Gaussian Noise)
- **Parameters**: Configurable SNR (Signal-to-Noise Ratio)
- **Implementation**: Add Gaussian noise to transmitted signals

### 3.2 Modulation Scheme
- **Initial Scheme**: BPSK (Binary Phase Shift Keying)
- **Rationale**: Simplest scheme for baseline implementation
- **Future Extensions**: QPSK, QAM can be added later

### 3.3 Relay Type
- **Initial Implementation**: AF (Amplify and Forward)
- **Description**: Classical relay technique
- **Operation**: 
  - Receive noisy signal from source
  - Amplify signal (with power normalization)
  - Forward to destination
- **Future Extensions**: Decode-and-Forward, GenAI-based relay

### 3.4 Performance Metrics
- **Primary Metric**: BER (Bit Error Rate) vs SNR
- **Validation**: Compare against theoretical AF relay performance
- **Visualization**: BER curves plotted using matplotlib

---

## 4. Documentation Requirements

### 4.1 Traceability
- Each implementation step must be traceable
- Ability to roll back to previous steps if current step fails
- Clear documentation of what changed between steps

### 4.2 Required Documentation Files

#### INPUT.md (This File)
- Master reference with all requirements
- Single source of truth for the project

#### IMPLEMENTATION_PLAN.md
- Overall strategy and architecture
- Module breakdown and responsibilities
- Step-by-step implementation guide
- Progress tracking with checkboxes

#### CHECKPOINT_LOG.md
- Detailed execution log for each checkpoint
- Test results and validation status
- Expected outputs vs actual outputs
- Rollback instructions for each checkpoint

#### CHANGELOG.md
- Detailed change tracking between checkpoints
- What was added/modified/deleted
- Reason for each change
- Impact on other components
- Code diff summaries

### 4.3 Session Recovery
- Documentation must support recovery from session termination
- Clear indication of last completed checkpoint
- Ability to continue from any checkpoint without starting over

---

## 5. Project Structure

```
relay_communication/
├── INPUT.md                        # This file - master requirements
├── IMPLEMENTATION_PLAN.md          # Overall plan and architecture
├── CHECKPOINT_LOG.md               # Execution log with test results
├── CHANGELOG.md                    # Change tracking between steps
├── checkpoints/                    # Incremental development files
│   ├── checkpoint_01_channel.py
│   ├── checkpoint_02_modulation.py
│   ├── checkpoint_03_nodes.py
│   └── checkpoint_04_simulation.py
├── tests/                          # Unit tests for each component
│   ├── test_channel.py
│   ├── test_modulation.py
│   └── test_nodes.py
└── final/                          # Final integrated version
    ├── channel.py
    ├── modulation.py
    ├── nodes.py
    └── simulation.py
```

---

## 6. Implementation Approach

### 6.1 Checkpoint-Based Development
- Each major component is a separate checkpoint
- Each checkpoint is independently testable
- Each checkpoint includes validation tests
- Progress tracked in all documentation files

### 6.2 Checkpoint Sequence
1. **Checkpoint 01**: AWGN Channel implementation and testing
2. **Checkpoint 02**: BPSK Modulation implementation and testing
3. **Checkpoint 03**: Source, Relay (AF), and Destination nodes
4. **Checkpoint 04**: Full simulation loop with BER calculation
5. **Checkpoint 05**: BER plot generation and performance validation

### 6.3 Testing Strategy
- Unit tests for each component
- Integration tests for combined components
- Validation against theoretical performance
- Visual verification through BER plots

---

## 7. Module Specifications

### 7.1 channel.py
**Purpose**: Implement wireless channel model

**Functions**:
- `awgn_channel(signal, snr_db)`: Add AWGN to signal
- Helper functions for SNR calculations

**Inputs**: Clean signal, SNR in dB
**Outputs**: Noisy signal

### 7.2 modulation.py
**Purpose**: Implement modulation/demodulation

**Functions**:
- `bpsk_modulate(bits)`: Map bits to BPSK symbols
- `bpsk_demodulate(symbols)`: Detect bits from symbols

**Inputs**: Bit stream or symbol stream
**Outputs**: Symbol stream or bit stream

### 7.3 nodes.py
**Purpose**: Implement communication nodes

**Classes**:
- `Source`: Generate and modulate bit streams
- `Relay`: Base class for relay implementations
- `AmplifyAndForwardRelay`: AF relay implementation
- `Destination`: Receive and demodulate signals

### 7.4 simulation.py
**Purpose**: Orchestrate the communication system

**Functions**:
- Monte Carlo simulation loop
- BER calculation
- SNR sweep
- Results collection

---

## 8. Dependencies

### Required Python Packages
- `numpy`: Numerical operations, signal processing
- `matplotlib`: Plotting BER curves
- `scipy` (optional): Statistical functions

### Installation
```bash
pip install numpy matplotlib scipy
```

---

## 9. Success Criteria

### Per Checkpoint
- [ ] Code runs without errors
- [ ] Unit tests pass
- [ ] Expected output matches actual output
- [ ] Documentation updated (CHECKPOINT_LOG.md, CHANGELOG.md)

### Overall Project
- [ ] Complete two-hop transmission works end-to-end
- [ ] BER vs SNR curve generated
- [ ] Performance matches theoretical AF relay behavior
- [ ] All documentation files complete and accurate
- [ ] Code is modular and ready for GenAI relay integration

---

## 10. Future Extensions

### Phase 2: Generative AI Integration
- Replace classical AF relay with GenAI-based relay
- Compare performance: Classical vs GenAI
- Explore different GenAI architectures
- Optimize for various channel conditions

### Potential GenAI Approaches
- Neural network-based signal processing
- Learned denoising and forwarding
- Adaptive relay strategies
- End-to-end learning

---

## 11. Notes and Constraints

- Start with simplest implementations (BPSK, AF)
- Prioritize modularity for easy component swapping
- Maintain clear separation between components
- Document all assumptions and design decisions
- Keep code readable and well-commented

---

## 12. Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-02-14 | Initial specification | Cline |

---

**End of Input Specification**
