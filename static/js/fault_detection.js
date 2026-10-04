/* ============================================================
   Fault Detection - Frontend (Stage 1-3)
   ============================================================ */

   const FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"];

   document.getElementById("predictForm").addEventListener("submit", async function (e) {
     e.preventDefault();
   
     const submitBtn = document.getElementById("submitBtn");
     const loading   = document.getElementById("loading");
     const errorBox  = document.getElementById("errorBox");
     const results   = document.getElementById("results");
   
     errorBox.classList.add("hidden");
     results.classList.add("hidden");
     ["stage2Box", "stage3Box"].forEach(id =>
       document.getElementById(id).classList.add("hidden")
     );
     resetStageClass("stage1Box");
     resetStageClass("stage2Box");
     resetStageClass("stage3Box");
   
     submitBtn.disabled = true;
     loading.classList.remove("hidden");
   
     const payload = {};
     for (const f of FEATURES) {
       payload[f] = document.getElementById(f).value;
     }
   
     try {
       const res = await fetch("/predict", {
         method: "POST",
         headers: { "Content-Type": "application/json" },
         body: JSON.stringify(payload),
       });
   
       const data = await res.json();
   
       if (!res.ok || data.error) {
         throw new Error(data.error || "Terjadi kesalahan pada server.");
       }
   
       renderResults(data.result);
     } catch (err) {
       errorBox.textContent = "❌ " + err.message;
       errorBox.classList.remove("hidden");
     } finally {
       submitBtn.disabled = false;
       loading.classList.add("hidden");
     }
   });
   
   document.getElementById("predictForm").addEventListener("reset", () => {
     document.getElementById("results").classList.add("hidden");
     document.getElementById("errorBox").classList.add("hidden");
     ["stage1Box", "stage2Box", "stage3Box"].forEach(resetStageClass);
   });
   
   function renderResults(r) {
     const results = document.getElementById("results");
   
     const s1Box = document.getElementById("stage1Box");
     document.getElementById("stage1Label").textContent = r.stage1.label.toUpperCase();
     document.getElementById("stage1Desc").textContent  = r.stage1.description;
     setConfidence("stage1Bar", "stage1Conf", r.stage1.confidence);
   
     if (r.stage1.label === "healthy") {
       s1Box.classList.add("success");
     } else {
       s1Box.classList.add("danger");
     }
   
     if (r.stage1.label === "healthy") {
       results.classList.remove("hidden");
       return;
     }
   
     if (r.stage2) {
       const s2Box = document.getElementById("stage2Box");
       s2Box.classList.remove("hidden");
       document.getElementById("stage2Label").textContent = r.stage2.label.toUpperCase();
       document.getElementById("stage2Desc").textContent  = r.stage2.description;
       setConfidence("stage2Bar", "stage2Conf", r.stage2.confidence);
       s2Box.classList.add(r.stage2.label === "symmetrical" ? "danger" : "warning");
     }
   
     if (r.stage3) {
       const s3Box = document.getElementById("stage3Box");
       s3Box.classList.remove("hidden");
       document.getElementById("stage3Label").textContent = r.stage3.label;
       document.getElementById("stage3Desc").textContent  = r.stage3.description;
       setConfidence("stage3Bar", "stage3Conf", r.stage3.confidence);
       s3Box.classList.add("info");
     }
   
     results.classList.remove("hidden");
   }
   
   function setConfidence(barId, confId, value) {
     const v = Number(value) || 0;
     document.getElementById(barId).style.width = v + "%";
     document.getElementById(confId).textContent = v.toFixed(2) + "%";
   }
   
   function resetStageClass(boxId) {
     const box = document.getElementById(boxId);
     if (!box) return;
     box.classList.remove("success", "warning", "danger", "info");
   }