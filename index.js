const Tesseract = require("tesseract.js");
const { JSDOM } = require("jsdom");
const { createCanvas, Image } = require("canvas");
const fs = require("node:fs");
const path = require("node:path");
const mangaPath = "manga.png";
const newPath = "new.png";

const mangaImage = fs.readFileSync(mangaPath);
const newImage = new Image();
newImage.src = fs.readFileSync(newPath);

Tesseract.recognize(mangaImage, "eng")
  .then(({ data: { hocr } }) => {
    const canvas = createCanvas(newImage.width, newImage.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(newImage, 0, 0);
    const { window } = new JSDOM();
    const parser = new window.DOMParser();
    const hocrDoc = parser.parseFromString(hocr, "text/html");
    const wordElements = hocrDoc.getElementsByClassName("ocrx_word");
    for (let i = 0; i < wordElements.length; i++) {
      const wordElement = wordElements[i];
      const title = wordElement.getAttribute("title");
      const bbox = title.match(/bbox (\d+) (\d+) (\d+) (\d+);/);
      if (bbox) {
        const x = parseInt(bbox[1], 10);
        const y = parseInt(bbox[2], 10);
        const width = parseInt(bbox[3], 10) - x;
        const height = parseInt(bbox[4], 10) - y;
        ctx.font = `${height}px serif`;
        ctx.fillStyle = "white";
        // background by 40 pixels (20 pixels on the left and 20 pixels on the right)
        // ctx.fillRect(x - 20, y, width + 40, height);
        ctx.fillRect(x, y, width, height);
        ctx.fillStyle = "black";
        ctx.fillText(wordElement.innerHTML, x, y + height);
      }
    }

    const stream = canvas.createPNGStream();
    const chunks = [];
    const exportPath = path.basename(newPath, path.extname(newPath)) + "_modified" + path.extname(newPath);
    stream.on("data", (chunk) => chunks.push(chunk));
    stream.on("end", () =>
      fs.writeFileSync(exportPath, Buffer.concat(chunks)),
      console.log("Done!!")
    );
  })
  .catch(function (error) {
    console.error(error);
  });
