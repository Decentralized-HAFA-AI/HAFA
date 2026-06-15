#!/bin/bash
echo "🚀 Starting HAFA Genesis Node..."
cargo run --release &
NODE_PID=$!

echo "⏳ Waiting 5 seconds for the node to initialize..."
sleep 5

echo "⛏️  Starting HAFA Miner..."
cargo run --bin hafa-miner --release &
MINER_PID=$!

echo "✅ Both Node and Miner are running!"
echo "Press Ctrl+C to stop both processes."

wait $NODE_PID $MINER_PID