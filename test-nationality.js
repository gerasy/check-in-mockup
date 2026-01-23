import { pipeline } from '@huggingface/transformers';
import { readdir, readFile } from 'fs/promises';
import { join, basename } from 'path';

// Test images directory
const FOTOS_DIR = '/Users/gerasym/Documents/projects/urbansports/fotos';

// Nationality labels for CLIP
const nationalityLabels = [
  'a photo of a Ukrainian person',
  'a photo of an Indian person',
  'a photo of a Syrian person',
  'a photo of a Georgian person',
  'a photo of an Albanian person',
  'a photo of a German person'
];

const labelToNationality = {
  'a photo of a Ukrainian person': 'ukraine',
  'a photo of an Indian person': 'india',
  'a photo of a Syrian person': 'syria',
  'a photo of a Georgian person': 'georgia',
  'a photo of an Albanian person': 'albania',
  'a photo of a German person': 'germany'
};

// Map folder names to expected nationality
const folderToNationality = {
  'albanian-male': 'albania',
  'georgian-male': 'georgia',
  'german-male': 'germany',
  'ukranian-male': 'ukraine',
  'indian-male': 'india',
  'syrian-male': 'syria'
};

async function loadImages() {
  const images = [];

  try {
    const folders = await readdir(FOTOS_DIR);

    for (const folder of folders) {
      if (folder.startsWith('.')) continue; // Skip hidden files

      const folderPath = join(FOTOS_DIR, folder);
      const expectedNationality = folderToNationality[folder];

      if (!expectedNationality) {
        console.log(`Skipping unknown folder: ${folder}`);
        continue;
      }

      try {
        const files = await readdir(folderPath);

        for (const file of files) {
          if (file.startsWith('.')) continue; // Skip hidden files
          if (!file.match(/\.(png|jpg|jpeg|webp)$/i)) continue; // Only images

          const filePath = join(folderPath, file);
          images.push({
            path: filePath,
            filename: file,
            folder: folder,
            expectedNationality: expectedNationality
          });
        }
      } catch (err) {
        console.error(`Error reading folder ${folder}:`, err.message);
      }
    }
  } catch (err) {
    console.error('Error reading fotos directory:', err.message);
  }

  return images;
}

async function main() {
  console.log('='.repeat(60));
  console.log('NATIONALITY CLASSIFIER TEST');
  console.log('='.repeat(60));
  console.log('');

  // Load images
  console.log('Loading test images...');
  const images = await loadImages();
  console.log(`Found ${images.length} test images\n`);

  if (images.length === 0) {
    console.log('No images found to test!');
    return;
  }

  // List images by folder
  console.log('Test images by nationality:');
  const byFolder = {};
  for (const img of images) {
    if (!byFolder[img.folder]) byFolder[img.folder] = [];
    byFolder[img.folder].push(img.filename);
  }
  for (const [folder, files] of Object.entries(byFolder)) {
    console.log(`  ${folder}: ${files.length} images`);
  }
  console.log('');

  // Load CLIP model
  console.log('Loading CLIP model (this may take a minute on first run)...');
  const startLoad = Date.now();
  const classifier = await pipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32');
  console.log(`Model loaded in ${((Date.now() - startLoad) / 1000).toFixed(1)}s\n`);

  // Test each image
  console.log('Running classification tests...\n');
  console.log('-'.repeat(60));

  const results = [];

  for (const img of images) {
    try {
      // Read image as base64
      const imageBuffer = await readFile(img.path);
      const base64 = imageBuffer.toString('base64');
      const mimeType = img.filename.toLowerCase().endsWith('.png') ? 'image/png' : 'image/jpeg';
      const dataUrl = `data:${mimeType};base64,${base64}`;

      // Run classification
      const output = await classifier(dataUrl, nationalityLabels);

      // Get top prediction
      const topResult = output[0];
      const predictedNationality = labelToNationality[topResult.label];
      const isCorrect = predictedNationality === img.expectedNationality;

      results.push({
        filename: img.filename,
        folder: img.folder,
        expected: img.expectedNationality,
        predicted: predictedNationality,
        confidence: topResult.score,
        correct: isCorrect,
        allScores: output
      });

      // Print result
      const status = isCorrect ? 'CORRECT' : 'WRONG';
      const statusIcon = isCorrect ? '\u2713' : '\u2717';
      console.log(`${statusIcon} ${img.folder}/${img.filename}`);
      console.log(`   Expected: ${img.expectedNationality.toUpperCase()}`);
      console.log(`   Predicted: ${predictedNationality.toUpperCase()} (${(topResult.score * 100).toFixed(1)}%)`);

      // Show top 3 predictions
      console.log('   All scores:');
      for (const r of output) {
        const nat = labelToNationality[r.label];
        const pct = (r.score * 100).toFixed(1);
        const marker = nat === img.expectedNationality ? ' <-- expected' : '';
        console.log(`      ${nat}: ${pct}%${marker}`);
      }
      console.log('');

    } catch (err) {
      console.error(`Error processing ${img.filename}:`, err.message);
      results.push({
        filename: img.filename,
        folder: img.folder,
        expected: img.expectedNationality,
        predicted: 'error',
        confidence: 0,
        correct: false,
        error: err.message
      });
    }
  }

  // Summary
  console.log('-'.repeat(60));
  console.log('\nSUMMARY');
  console.log('='.repeat(60));

  const correct = results.filter(r => r.correct).length;
  const total = results.length;
  const accuracy = ((correct / total) * 100).toFixed(1);

  console.log(`\nOverall Accuracy: ${correct}/${total} (${accuracy}%)\n`);

  // Per-nationality breakdown
  console.log('Per-Nationality Results:');
  const byNationality = {};
  for (const r of results) {
    if (!byNationality[r.expected]) {
      byNationality[r.expected] = { correct: 0, total: 0, predictions: {} };
    }
    byNationality[r.expected].total++;
    if (r.correct) byNationality[r.expected].correct++;

    // Track what it was predicted as
    if (!byNationality[r.expected].predictions[r.predicted]) {
      byNationality[r.expected].predictions[r.predicted] = 0;
    }
    byNationality[r.expected].predictions[r.predicted]++;
  }

  for (const [nat, data] of Object.entries(byNationality)) {
    const pct = ((data.correct / data.total) * 100).toFixed(0);
    console.log(`\n  ${nat.toUpperCase()}: ${data.correct}/${data.total} (${pct}%)`);
    console.log(`    Predicted as:`);
    for (const [pred, count] of Object.entries(data.predictions)) {
      const marker = pred === nat ? ' (correct)' : '';
      console.log(`      ${pred}: ${count}${marker}`);
    }
  }

  console.log('\n' + '='.repeat(60));
  console.log('TEST COMPLETE');
  console.log('='.repeat(60));
}

main().catch(console.error);
