#!/bin/bash
echo "=== MCP Server Verification ==="
echo ""

# Depwire
echo "1. Depwire:"
if command -v depwire &>/dev/null; then
    echo "   ✓ Installed (version: $(depwire --version 2>&1 | head -1))"
    health=$(depwire health . 2>&1 | head -5)
    echo "   Health score: $(echo "$health" | grep -E 'Overall:[[:space:]]*[0-9]+')"
else
    echo "   ✗ Not found"
fi
echo ""

# Open-WebSearch
echo "2. Open-WebSearch:"
if curl -s http://localhost:3000/ 2>&1 | grep -q "Cannot GET"; then
    echo "   ✓ Running on http://localhost:3000"
else
    echo "   ✗ Not responding"
fi
echo ""

# Hindsight
echo "3. Hindsight (Ollama):"
if curl -s http://localhost:8888/health 2>&1 | grep -q "healthy"; then
    echo "   ✓ Running on http://localhost:8888"
    echo "   ✓ MCP endpoint: http://localhost:8888/mcp"
else
    echo "   ✗ Not responding"
fi
echo ""

# Firecrawl
echo "4. Firecrawl:"
if docker ps | grep -q firecrawl-api; then
    echo "   ✓ Docker containers running"
    echo "   ✓ API endpoint: http://localhost:3002"
    # Test curl if possible
    if curl -s -X POST http://localhost:3002/v1/crawl -H 'Content-Type: application/json' -d '{"url":"https://example.com"}' 2>&1 | grep -q '"success":true"'; then
        echo "   ✓ Crawl API responding"
    fi
else
    echo "   ✗ Docker containers not running"
fi
echo ""

# MCP config
echo "5. Kilo Code MCP Config:"
if grep -q '"depwire"' /home/phhofm/.config/Code/User/mcp.json; then
    echo "   ✓ Depwire configured"
fi
if grep -q '"open-websearch"' /home/phhofm/.config/Code/User/mcp.json; then
    echo "   ✓ Open-WebSearch configured"
fi
if grep -q '"hindsight"' /home/phhofm/.config/Code/User/mcp.json; then
    echo "   ✓ Hindsight configured"
fi
if grep -q '"firecrawl"' /home/phhofm/.config/Code/User/mcp.json; then
    echo "   ✓ Firecrawl configured"
fi

echo ""
echo "=== All Done! ==="
echo "Restart VSCode/Kilo Code for MCP servers to connect."
