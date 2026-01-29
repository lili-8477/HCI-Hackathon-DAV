# Data Analysis & Visualization Assistant - Access Guide

## 🎉 Deployment Status: SUCCESS

The application is successfully deployed and running on the CHPC cluster!

---

## 🔒 Important: CHPC Cluster Network Security

The CHPC cluster has firewall rules that prevent direct access to compute nodes from external networks. To access the application, you need to use **SSH port forwarding**.

---

## 📍 Access Methods

### Method 1: SSH Port Forwarding (Recommended for Demo)

This creates a secure tunnel from your local machine to the CHPC cluster.

#### On Your Local Computer (Mac/Linux):
```bash
ssh -L 8501:10.242.16.202:8501 u6025146@notchpeak.chpc.utah.edu
```

#### On Windows (PowerShell):
```powershell
ssh -L 8501:10.242.16.202:8501 u6025146@notchpeak.chpc.utah.edu
```

Then open in your browser:
- **http://localhost:8501**

#### For Demo Participants:
Each person who wants to access the app needs to:
1. Have SSH access to CHPC
2. Run the SSH port forwarding command above
3. Access http://localhost:8501 in their browser

---

### Method 2: Direct Access (Only from CHPC Login/Compute Nodes)

If you're already on a CHPC login node or compute node, you can access directly:
- **http://10.242.16.202:8501**
- **http://localhost:8501** (if on the same node)

---

### Method 3: Interactive Notebook Session

If you're running this through OnDemand or JupyterHub on CHPC, you can access:
- **http://10.242.16.202:8501** (from within the CHPC environment)

---

## 🚀 Quick Demo Setup

### For a Single-User Demo:

1. **On your laptop**, open a terminal and run:
   ```bash
   ssh -L 8501:10.242.16.202:8501 u6025146@notchpeak.chpc.utah.edu
   ```

2. **Keep the SSH session open** (it creates the tunnel)

3. **Open your browser** to http://localhost:8501

4. **Demo the application** - it will be running on your laptop's localhost but powered by the CHPC cluster

### For Multi-User Demo on Campus:

**Option A: Screen Sharing**
- Set up SSH forwarding on your laptop
- Access the app at http://localhost:8501
- Share your screen with demo participants

**Option B: Public Tunnel (ngrok/cloudflared)**
- Requires ngrok account authentication OR
- Use cloudflared for temporary public URL
- Best for larger group demos

**Option C: CHPC OnDemand**
- Access via CHPC's Open OnDemand portal
- Run the app through a VNC or JupyterHub session
- Share screen or provide temporary access

---

## 🔧 Technical Details

### Services Running
1. **Ollama Server**
   - Status: ✓ Running
   - Model: qwen3:8b (5.2 GB)
   - Port: 11434
   - Process ID: 278638

2. **Streamlit Application**
   - Status: ✓ Running
   - Port: 8501
   - Binding: 0.0.0.0 (all network interfaces)
   - Process ID: 280113
   - Log: `/tmp/streamlit.log`

### Network Configuration
- **Primary IP**: 10.242.16.202 (eth0)
- **InfiniBand IP**: 10.242.76.202 (ib0)
- **Listening Port**: 8501 (TCP)
- **Firewall**: CHPC cluster firewall restricts external access

---

## 📊 Using the Application

1. **Upload a dataset** (CSV, Excel, or JSON)
2. **Ask questions** in natural language
3. **Get AI-powered insights** with visualizations

### Sample Datasets
- `data/sample_sales.csv` - E-commerce sales data
- `data/sample_iris.csv` - Classic Iris dataset

### Example Queries
- "Show me summary statistics"
- "Plot a correlation heatmap"
- "Show the distribution of revenue"
- "Create a scatter plot of price vs quantity"
- "Which region has the highest sales?"

---

## 🛑 Managing the Application

### Check if Services are Running
```bash
ps aux | grep streamlit
ps aux | grep ollama
```

### Stop Services
```bash
# Stop Streamlit
pkill -f "streamlit run app.py"

# Stop Ollama
pkill -f "ollama serve"
```

### Restart Services
```bash
cd /uufs/chpc.utah.edu/common/home/jonesk-group2/HCI-Hackathon/HCI-Hackathon-DAV

# Start Ollama
module load ollama/0.12.5
export OLLAMA_MODELS=$HOME/.ollama/models
nohup ollama serve > /tmp/ollama.log 2>&1 &

# Wait a few seconds, then start Streamlit
sleep 3
source venv/bin/activate
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > /tmp/streamlit.log 2>&1 &
```

### View Logs
```bash
# Streamlit logs
tail -f /tmp/streamlit.log

# Ollama logs
tail -f /tmp/ollama.log
```

---

## 🌐 Alternative: Set Up Public Access (Optional)

If you need a public URL for a demo to people without CHPC access:

### Using cloudflared (No registration required)
```bash
# Download cloudflared
wget -O /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /tmp/cloudflared

# Create tunnel
/tmp/cloudflared tunnel --url http://localhost:8501
```

This will generate a temporary public URL like: `https://xxx.trycloudflare.com`

### Using ngrok (Requires free account)
```bash
# Sign up at https://dashboard.ngrok.com/signup
# Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken

# Configure authtoken
$HOME/.local/bin/ngrok authtoken YOUR_TOKEN_HERE

# Start tunnel
$HOME/.local/bin/ngrok http 8501
```

---

## 📝 Troubleshooting

### "Connection refused" Error
- **Cause**: Trying to access from outside CHPC network without SSH tunnel
- **Solution**: Use SSH port forwarding (see Method 1 above)

### Application Not Loading
1. Check if services are running: `ps aux | grep streamlit`
2. Check logs: `tail -20 /tmp/streamlit.log`
3. Verify port is listening: `netstat -tlnp | grep 8501`

### SSH Tunnel Not Working
- Ensure you're using the correct username and hostname
- Make sure your SSH key is configured for CHPC
- Check that you don't have a firewall blocking SSH connections

---

**Deployed**: 2026-01-29  
**Location**: CHPC Notchpeak Cluster  
**Node**: 10.242.16.202  
**Port**: 8501  
**Access**: SSH Port Forwarding Required
