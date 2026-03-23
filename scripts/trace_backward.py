import torch
import torch.nn as nn

torch.manual_seed(0)

EMB_ROWS, EMB_DIM = 8, 3
HIDDEN = 4
BATCH = 2
NUM_TABLES = 2


class TinyDLRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tables = nn.ModuleList(
            [nn.Embedding(EMB_ROWS, EMB_DIM) for _ in range(NUM_TABLES)]
        )
        self.linear1 = nn.Linear(NUM_TABLES * EMB_DIM, HIDDEN)
        self.linear2 = nn.Linear(HIDDEN, 1)

    def forward(self, indices):
        embs = [self.tables[i](indices[i]) for i in range(NUM_TABLES)]
        cat = torch.cat(embs, dim=1)
        z1 = self.linear1(cat)
        a1 = torch.relu(z1)
        logit = self.linear2(a1)
        return logit


model = TinyDLRM()
indices = [torch.tensor([2, 5]), torch.tensor([0, 7])]
labels = torch.tensor([[1.0], [0.0]])

# ============================================================
# Inspect the graph that autograd builds during forward
# ============================================================
logits = model(indices)
loss_for_graph = nn.functional.binary_cross_entropy_with_logits(logits, labels)

print("=" * 60)
print("COMPUTATION GRAPH (grad_fn chain built during forward)")
print("=" * 60)
node = loss_for_graph.grad_fn
queue = [(node, 0)]
visited = set()
while queue:
    n, depth = queue.pop(0)
    if id(n) in visited or n is None:
        continue
    visited.add(id(n))
    print(f"{'  ' * depth}↳ {n.__class__.__name__}")
    for child, _ in n.next_functions:
        if child is not None:
            queue.append((child, depth + 1))
print()

# ============================================================
# Re-run forward saving intermediates for manual backward
# ============================================================
model.zero_grad()

embs = [model.tables[i](indices[i]) for i in range(NUM_TABLES)]
cat = torch.cat(embs, dim=1)
z1 = model.linear1(cat)
a1 = torch.relu(z1)
logit = model.linear2(a1)
loss = nn.functional.binary_cross_entropy_with_logits(logit, labels)

W1, b1 = model.linear1.weight, model.linear1.bias
W2, b2 = model.linear2.weight, model.linear2.bias

print("=" * 60)
print("FORWARD VALUES")
print("=" * 60)
print(f"emb0 (table0[{indices[0].tolist()}]):\n{embs[0].detach()}")
print(f"emb1 (table1[{indices[1].tolist()}]):\n{embs[1].detach()}")
print(f"cat:   {cat.shape}\n{cat.detach()}")
print(f"z1:    {z1.shape}\n{z1.detach()}")
print(f"a1:    {a1.shape}\n{a1.detach()}")
print(f"logit: {logit.shape} = {logit.detach().flatten().tolist()}")
print(f"loss:  {loss.item():.6f}")
print()

# ============================================================
# MANUAL BACKWARD — step by step
# ============================================================
print("=" * 60)
print("MANUAL BACKWARD (replicating the autograd engine)")
print("=" * 60)
print()

# The engine processes nodes in reverse topological order.
# It maintains a dependency count and a ready queue.
print("Engine state:")
print("  ready_queue = [BinaryCrossEntropyWithLogitsBackward]")
print("  dependencies = {MmBackward(L2): 1, ReluBackward: 1,")
print("                   MmBackward(L1): 1, CatBackward: 1,")
print("                   EmbeddingBackward0: 1, EmbeddingBackward1: 1,")
print("                   AccumulateGrad(W2): 1, AccumulateGrad(b2): 1,")
print("                   AccumulateGrad(W1): 1, AccumulateGrad(b1): 1,")
print("                   AccumulateGrad(T0): 1, AccumulateGrad(T1): 1}")
print()

# --- Node 1: BCEWithLogitsBackward ---
print("─" * 60)
print("POP: BinaryCrossEntropyWithLogitsBackward")
print("  input:  grad_output = tensor(1.)  [seeded by engine]")
pred = torch.sigmoid(logit.detach())
dlogit = (pred - labels) / BATCH
print(f"  apply:  sigmoid(logit) = {pred.flatten().tolist()}")
print(f"  apply:  dL/dlogit = (pred - labels) / B = {dlogit.flatten().tolist()}")
print(f"  output: send dL/dlogit → MmBackward(linear2)")
print(f"  deps:   MmBackward(L2): 1→0, ENQUEUE")
print()

# --- Node 2: MmBackward for linear2 ---
print("─" * 60)
print("POP: MmBackward (linear2: logit = a1 @ W2^T + b2)")
print(f"  input:  dL/dlogit = {dlogit.detach().flatten().tolist()}")
da1 = dlogit @ W2.detach()
dW2_manual = dlogit.t() @ a1.detach()
db2_manual = dlogit.sum(dim=0)
print(f"  apply:  dL/da1 = dlogit @ W2          = shape {da1.shape}")
print(f"          dL/dW2 = dlogit^T @ a1         = shape {dW2_manual.shape}")
print(f"          dL/db2 = sum(dlogit)            = {db2_manual.detach().flatten().tolist()}")
print(f"  output: send dL/da1 → ReluBackward")
print(f"          send dL/dW2 → AccumulateGrad(W2)")
print(f"          send dL/db2 → AccumulateGrad(b2)")
print(f"  deps:   ReluBackward: 1→0, ENQUEUE")
print(f"          AccumulateGrad(W2): 1→0, ENQUEUE")
print(f"          AccumulateGrad(b2): 1→0, ENQUEUE")
print()

# --- Node 2b: AccumulateGrad for W2, b2 ---
print("─" * 60)
print("POP: AccumulateGrad(W2)")
print(f"  W2.grad += dL/dW2   (leaf node, stores gradient)")
print("POP: AccumulateGrad(b2)")
print(f"  b2.grad += dL/db2   (leaf node, stores gradient)")
print()

# --- Node 3: ReluBackward ---
print("─" * 60)
print("POP: ReluBackward (a1 = relu(z1))")
relu_mask = (z1.detach() > 0).float()
dz1 = da1 * relu_mask
print(f"  input:  dL/da1")
print(f"  saved:  result a1 (to compute mask = a1 > 0)")
print(f"  apply:  mask = {relu_mask.flatten().tolist()}")
print(f"          dL/dz1 = dL/da1 * mask")
print(f"  output: send dL/dz1 → MmBackward(linear1)")
print(f"  deps:   MmBackward(L1): 1→0, ENQUEUE")
print()

# --- Node 4: MmBackward for linear1 ---
print("─" * 60)
print("POP: MmBackward (linear1: z1 = cat @ W1^T + b1)")
dcat = dz1 @ W1.detach()
dW1_manual = dz1.t() @ cat.detach()
db1_manual = dz1.sum(dim=0)
print(f"  input:  dL/dz1")
print(f"  saved:  cat (forward input) and W1")
print(f"  apply:  dL/dcat = dz1 @ W1             = shape {dcat.shape}")
print(f"          dL/dW1 = dz1^T @ cat            = shape {dW1_manual.shape}")
print(f"          dL/db1 = sum(dz1)               = {db1_manual.detach().flatten().tolist()}")
print(f"  output: send dL/dcat → CatBackward")
print(f"          send dL/dW1 → AccumulateGrad(W1)")
print(f"          send dL/db1 → AccumulateGrad(b1)")
print(f"  deps:   CatBackward: 1→0, ENQUEUE")
print()

# --- Node 4b: AccumulateGrad for W1, b1 ---
print("─" * 60)
print("POP: AccumulateGrad(W1)")
print(f"  W1.grad += dL/dW1   (leaf node)")
print("POP: AccumulateGrad(b1)")
print(f"  b1.grad += dL/db1   (leaf node)")
print()

# --- Node 5: CatBackward ---
print("─" * 60)
print("POP: CatBackward (cat = concat(emb0, emb1) along dim=1)")
demb0, demb1 = dcat.split(EMB_DIM, dim=1)
print(f"  input:  dL/dcat shape {dcat.shape}")
print(f"  apply:  split along dim=1 into chunks of size {EMB_DIM}")
print(f"          dL/demb0 shape {demb0.shape}")
print(f"          dL/demb1 shape {demb1.shape}")
print(f"  output: send dL/demb0 → EmbeddingBackward(table0)")
print(f"          send dL/demb1 → EmbeddingBackward(table1)")
print(f"  deps:   EmbeddingBackward0: 1→0, ENQUEUE")
print(f"          EmbeddingBackward1: 1→0, ENQUEUE")
print()

# --- Node 6: EmbeddingBackward ---
for t in range(NUM_TABLES):
    print("─" * 60)
    demb = [demb0, demb1][t]
    print(f"POP: EmbeddingBackward (table {t})")
    print(f"  input:  dL/demb{t} shape {demb.shape}")
    print(f"  saved:  indices = {indices[t].tolist()}")
    print(f"  apply:  SPARSE scatter-add (no matmul!)")
    grad_weight = torch.zeros(EMB_ROWS, EMB_DIM)
    for b in range(BATCH):
        row = indices[t][b].item()
        grad_weight[row] += demb[b].detach()
        print(f"          table.weight.grad[{row}] += {demb[b].detach().tolist()}")
    print(f"  output: send sparse grad → AccumulateGrad(table{t}.weight)")
    print(f"          table{t}.weight.grad =")
    for r in range(EMB_ROWS):
        if grad_weight[r].abs().sum() > 0:
            print(f"            row {r}: {grad_weight[r].tolist()}")
        else:
            print(f"            row {r}: [0, 0, 0]  (untouched)")
    print()

print("─" * 60)
print("QUEUE EMPTY — backward complete, graph destroyed.")
print()

# ============================================================
# VERIFY
# ============================================================
print("=" * 60)
print("VERIFICATION: manual gradients == autograd gradients")
print("=" * 60)
loss.backward()

for name, manual, auto in [
    ("W2", dW2_manual, W2.grad),
    ("b2", db2_manual, b2.grad),
    ("W1", dW1_manual, W1.grad),
    ("b1", db1_manual, b1.grad),
]:
    match = torch.allclose(manual.detach(), auto, atol=1e-6)
    print(f"  dL/d{name}: {'✓ MATCH' if match else '✗ MISMATCH'}")

for t in range(NUM_TABLES):
    demb = [demb0, demb1][t]
    auto_grad = model.tables[t].weight.grad
    manual_grad = torch.zeros_like(auto_grad)
    for b in range(BATCH):
        manual_grad[indices[t][b]] += demb[b].detach()
    match = torch.allclose(manual_grad, auto_grad, atol=1e-6)
    print(f"  dL/dTable{t}: {'✓ MATCH' if match else '✗ MISMATCH'}")
