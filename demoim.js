const { Translate } = require('@google-cloud/translate').v2;
const vision = require('@google-cloud/vision');
const { createCanvas, loadImage, registerFont } = require('canvas');
const axios = require('axios');
const Jimp = require('jimp');
const fs = require('fs');

const apiKey = '';
const imagePath = 'downloaded_image.png';
const serviceAccountPath = 'service.json';
registerFont(('arial.ttf'), { family: 'Arial Unicode MS' });
registerFont(('tahoma.ttf'), { family: 'Tahoma' });
registerFont(('NotoSansSC-VariableFont_wght.ttf'), { family: 'Noto Sans CJK SC' });
registerFont(('NotoSansJP-VariableFont_wght.ttf'), { family: 'Noto Sans CJK JP' });
registerFont(('NotoSansKR-VariableFont_wght.ttf'), { family: 'Noto Sans CJK KR' });

  async function translateimages(imageUrl, targetLang) {
    await downloadImage(imageUrl, imagePath);

    let jimpImage = await Jimp.read(imagePath);
    const textData = await picToText(imagePath);
    const words = extractWords(textData);
    const mergedWords = mergeWords(words);

    for (const word of mergedWords) {
        if (!isNumberOrSymbolOrSingleChar(word.text)) {
            await removeTextWithJimp(jimpImage, word);
        }
    }

    const editedBuffer = await jimpImage.getBufferAsync(Jimp.MIME_PNG);
    const img = await loadImage(editedBuffer);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    for (const word of mergedWords) {
        if (!isNumberOrSymbolOrSingleChar(word.text)) {
            await drawTranslatedText(ctx, jimpImage, word, targetLang);
        }
    }

    return new Promise((resolve, reject) => {
        canvas.toBuffer((err, buffer) => {
            if (err) {
                reject(err);
            } else {
                resolve(buffer);
            }
        });
    });
}

async function downloadImage(url, path) {
    const response = await axios({
        url,
        responseType: 'stream',
    });
    return new Promise((resolve, reject) => {
        const writer = fs.createWriteStream(path);
        response.data.pipe(writer);
        writer.on('finish', resolve);
        writer.on('error', reject);
    });
}

async function picToText(inputFile) {
    const client = new vision.ImageAnnotatorClient({
        keyFilename: serviceAccountPath,
    });
    const [result] = await client.textDetection({
        image: { source: { filename: inputFile } }
    });
    return result.fullTextAnnotation.pages.flatMap(page =>
        page.blocks.flatMap(block =>
            block.paragraphs.flatMap(paragraph =>
                paragraph.words.map(word => ({
                    text: word.symbols.map(symbol => symbol.text).join(''),
                    bbox: word.boundingBox
                }))
            )
        )
    );
}

function calculateRotationAngle(vertices) {
    const dx = vertices[1].x - vertices[0].x;
    const dy = vertices[1].y - vertices[0].y;
    const angle = Math.atan2(dy, dx) * (180 / Math.PI);
    return angle;
}

function extractWords(words) {
    return words.map(word => ({
        text: word.text,
        bbox: {
            x0: word.bbox.vertices[0].x,
            y0: word.bbox.vertices[0].y,
            x1: word.bbox.vertices[2].x,
            y1: word.bbox.vertices[2].y,
        },
        angle: calculateRotationAngle(word.bbox.vertices)
    }));
}

function isSkewed(angle, threshold = 10) {
    return Math.abs(angle) > threshold;
}

function mergeWords(words) {
    const mergedWords = [];
    let currentLine = [];
    let currentY = null;

    for (const word of words) {
        if (isSkewed(word.angle)) {
            continue;
        }

        if (currentY === null || Math.abs(word.bbox.y0 - currentY) <= 10) {
            if (currentLine.length === 0 || shouldCombine(currentLine[currentLine.length - 1], word)) {
                currentLine.push(word);
                currentY = word.bbox.y0;
            } else {
                mergedWords.push(combineLine(currentLine));
                currentLine = [word];
                currentY = word.bbox.y0;
            }
        } else {
            mergedWords.push(combineLine(currentLine));
            currentLine = [word];
            currentY = word.bbox.y0;
        }
    }

    if (currentLine.length > 0) {
        mergedWords.push(combineLine(currentLine));
    }

    return mergedWords;
}

function shouldCombine(word1, word2) {
    const isNumberOrSymbolOrSingleChar1 = isNumberOrSymbolOrSingleChar(word1.text);
    const isNumberOrSymbolOrSingleChar2 = isNumberOrSymbolOrSingleChar(word2.text);
    if (isNumberOrSymbolOrSingleChar1 !== isNumberOrSymbolOrSingleChar2) {
        return false;
    }

    const gap = word2.bbox.x0 - word1.bbox.x1;
    return gap >= 0 && gap <= 10;
}

function isNumberOrSymbolOrSingleChar(text) {
    const symbols = ['%', '+', '!', '@', '#', '$', '&', '*', '(', ')', '=', '{', '}', '[', ']', ';', ':', '<', '>', ',', '.', '?', '/', '|', '\\', '^', '~', '`'];
    const thaiRegex = /^[\u0E00-\u0E7F]+$/;
    const cjkRegex = /^[\u4E00-\u9FFF\u3400-\u4DBF\uAC00-\uD7AF]+$/;

    // if (!isNaN(text)) { // || symbols.some(symbol => text.includes(symbol))
    //     return true;
    // }

    if (thaiRegex.test(text) && text.length === 1) {
        return true;
    }

    if (cjkRegex.test(text) && text.length === 1) {
        return false;
    }

    if (thaiRegex.test(text) && text.length > 1) {
        return false;
    }

    if (text.length === 1) {
        return true;
    }

    return false;
}


function combineLine(line) {
    const text = line.map(word => word.text).join(' ');
    const x0 = Math.min(...line.map(word => word.bbox.x0));
    const y0 = Math.min(...line.map(word => word.bbox.y0));
    const x1 = Math.max(...line.map(word => word.bbox.x1));
    const y1 = Math.max(...line.map(word => word.bbox.y1));
    return { text, bbox: { x0, y0, x1, y1 } };
}

async function removeTextWithJimp(image, word) {
    const { x0, x1, y0, y1 } = word.bbox;
    const width = x1 - x0;
    const height = y1 - y0;

    const margin = 2;

    const imgWidth = image.bitmap.width;
    const imgHeight = image.bitmap.height;
    const validX0 = Math.max(0, Math.min(x0 - margin, imgWidth - 1));
    const validY0 = Math.max(0, Math.min(y0 - margin, imgHeight - 1));
    const validX1 = Math.max(0, Math.min(x1 + margin, imgWidth));
    const validY1 = Math.max(0, Math.min(y1 + margin, imgHeight));
    const validWidth = validX1 - validX0;
    const validHeight = validY1 - validY0;

    const region = image.clone().crop(validX0, validY0, validWidth, validHeight).blur(5);
    image.blit(region, validX0, validY0);
}

async function drawTranslatedText(ctx, jimpImage, word, targetLang) {
    const { x0, x1, y0, y1 } = word.bbox;
    const width = x1 - x0;
    const height = y1 - y0;

    const translatedText = await translateText(word.text, targetLang);
    const avgColor = await getAverageColor(jimpImage, x0, y0, width, height);
    const textColor = getContrastingColor(avgColor.r, avgColor.g, avgColor.b);

    ctx.fillStyle = textColor;
    ctx.font = `${height}px "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK KR", "Tahoma", "Arial Unicode MS", sans-serif`;
    const textMetrics = ctx.measureText(translatedText);

    if (textMetrics.width > width) {
        const scaleFactor = width / textMetrics.width;
        ctx.font = `${height * scaleFactor}px "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK KR", "Tahoma", "Arial Unicode MS", sans-serif`;
    }

    ctx.fillText(translatedText, x0, y1 - (height * 0.1)); 
}

async function getAverageColor(image, x, y, width, height) {
    let r = 0, g = 0, b = 0, count = 0;

    image.scan(x, y, width, height, function(x, y, idx) {
        r += this.bitmap.data[idx];
        g += this.bitmap.data[idx + 1];
        b += this.bitmap.data[idx + 2];
        count++;
    });

    r = Math.floor(r / count);
    g = Math.floor(g / count);
    b = Math.floor(b / count);

    return { r, g, b };
}

function getContrastingColor(r, g, b) {
    const yiq = ((r * 299) + (g * 587) + (b * 114)) / 1000;
    return (yiq >= 128) ? 'black' : 'white';
}

async function translateText(text, targetLang) {
    const translate = new Translate({ key: apiKey });
    const [translation] = await translate.translate(text, targetLang);
    return translation;
}

translateimages("url", "en").catch(console.error);
