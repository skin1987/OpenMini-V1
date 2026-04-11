name: Pull Request Template
about: Use this template when submitting a PR
labels: ['PR']
body: |
  ## PR Summary
  - [ ] Bug fix (non-breaking)
  - [ ] New feature (non-breaking)
  - [ ] Breaking change
  - [ ] Documentation update
  - [ ] Code style update
  - [ ] Performance improvement
  - [ ] Test coverage improvement
  
  ## Changes Made
  | File | Change Type | Description |
  |------|-------------|-------------|
  
  ## Testing
  - [x] Tests added/updated
  - [x] `cargo test --workspace --lib` passes
  - [x] `cargo clippy` passes (0 errors)
  - [x] `cargo fmt --check` passes
  
  ## Checklist
  - [x] Code follows project style
  - [x] Self-review completed
  - [x] Comments in hard-to-understand areas
  - [x] No new warnings introduced
