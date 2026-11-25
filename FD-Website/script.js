const inference = [
  {name:"MCNN", auc:0.830788, f1:0.695988, ap:0.523499},
  {name:"STAN", auc:0.863385, f1:0.686996, ap:0.659499},
  {name:"STAGN", auc:0.798717, f1:0.499737, ap:0.521715},
  {name:"GTAN", auc:0.804241, f1:0.735732, ap:0.623051},
  {name:"RGTAN", auc:0.691860, f1:0.452312, ap:0.356395}
];

const menuToggle = document.getElementById("menuToggle");
const nav = document.getElementById("mainNav");
menuToggle.addEventListener("click", () => nav.classList.toggle("open"));
nav.querySelectorAll("a").forEach(link => link.addEventListener("click", () => nav.classList.remove("open")));

const bars = document.getElementById("dashboardBars");
inference.forEach(item => {
  const row = document.createElement("div");
  row.className = "mini-row";
  row.innerHTML = `<span>${item.name}</span><div class="mini-track"><div class="mini-fill" style="width:${item.auc*100}%"></div></div><b>${item.auc.toFixed(3)}</b>`;
  bars.appendChild(row);
});

function drawBarChart(canvasId, metric, label) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth, height = canvas.clientHeight;
  canvas.width = width * dpr; canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
  const pad = {left:42, right:14, top:18, bottom:48};
  const plotW = width-pad.left-pad.right, plotH = height-pad.top-pad.bottom;
  ctx.clearRect(0,0,width,height);
  ctx.font = "11px DM Sans, sans-serif";
  ctx.strokeStyle = "#2a3d56"; ctx.fillStyle = "#9aabc2"; ctx.lineWidth = 1;
  for(let i=0;i<=5;i++){
    const y=pad.top+plotH-(i/5)*plotH;
    ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(width-pad.right,y);ctx.stroke();
    ctx.fillText((i/5).toFixed(1),7,y+4);
  }
  const slot=plotW/inference.length, bw=Math.min(36,slot*.55);
  inference.forEach((item,i)=>{
    const val=item[metric], x=pad.left+i*slot+(slot-bw)/2, h=val*plotH, y=pad.top+plotH-h;
    const grad=ctx.createLinearGradient(0,y,0,pad.top+plotH);
    grad.addColorStop(0,"#4ed6d0");grad.addColorStop(1,"#477fe8");
    ctx.fillStyle=grad;ctx.beginPath();ctx.roundRect(x,y,bw,h,5);ctx.fill();
    ctx.fillStyle="#dce8f7";ctx.textAlign="center";ctx.fillText(val.toFixed(3),x+bw/2,y-7);
    ctx.fillStyle="#9aabc2";ctx.fillText(item.name,x+bw/2,height-19);
  });
  ctx.textAlign="left";
}
function drawAll(){drawBarChart("aucChart","auc","AUC");drawBarChart("f1Chart","f1","F1");drawBarChart("apChart","ap","AP");}
window.addEventListener("load",drawAll);
window.addEventListener("resize",drawAll);

document.getElementById("transactionForm").addEventListener("submit", event => {
  event.preventDefault();
  const amount = Number(document.getElementById("amount").value);
  const type = document.getElementById("type").value;
  const source = document.getElementById("source").value.trim();
  const target = document.getElementById("target").value.trim();
  const location = document.getElementById("location").value.trim() || "Not provided";
  const panel = document.getElementById("predictionPanel");
  panel.innerHTML = `<div class="prediction-icon">✓</div><span class="tag">INPUT CAPTURED · DEMO ONLY</span><h3>Transaction details received</h3><p>Your demonstration input has been captured. This static website is not connected to a trained fraud-detection model, so it cannot determine whether this transaction is fraudulent.</p><div class="prediction-placeholder"><b>Amount:</b> ₹${amount.toLocaleString("en-IN",{minimumFractionDigits:2})}<br><b>Type:</b> ${type}<br><b>Sender:</b> ${escapeHtml(source)}<br><b>Receiver:</b> ${escapeHtml(target)}<br><b>Location:</b> ${escapeHtml(location)}<br><br><b>Prediction:</b> Not generated — backend model connection required.</div>`;
});
function escapeHtml(value){return value.replace(/[&<>"']/g, char=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[char]));}
