import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Minimal BirdBox ONNX host. WAV in, Raven Selection Table out.
 *
 * The ONNX graph already contains PCEN, YOLO, NMS, and song merge.
 * This file only loads audio, runs ONNX Runtime, and writes the same
 * table as detect_birds.py --output-format raven-selection-table.
 *
 * Setup (once), from the repo root:
 *
 *   curl -L -o tests/java/onnxruntime.jar \
 *     https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime/1.22.0/onnxruntime-1.22.0.jar
 *
 * Run (JDK 8+), from the repo root:
 *
 *   javac -cp tests/java/onnxruntime.jar tests/java/test_model.java
 *   java -cp tests/java:tests/java/onnxruntime.jar test_model
 *   java -cp tests/java:tests/java/onnxruntime.jar test_model tests/test.wav
 *   java -cp tests/java:tests/java/onnxruntime.jar test_model audio.wav model.onnx out.txt
 *   java -cp tests/java:tests/java/onnxruntime.jar test_model audio.wav model.onnx out.txt 0.18 0.1
 *
 * Args: [wav] [onnx] [out.txt] [conf] [song_gap]
 * Defaults: tests/test.wav, docs/models/Just-Bird_fp32.onnx,
 * tests/java/raven_selection_table.txt, plus conf / song_gap from ONNX metadata.
 *
 * Audio is mixed to mono and resampled to 32 kHz. The graph has no resampler.
 */
public class test_model {

    static final int SAMPLE_RATE = 32000;
    static final String RAVEN_HEADER =
            "Selection\tView\tChannel\tBegin Time (S)\tEnd Time (S)\tLow Freq (Hz)\tHigh Freq (Hz)\tAnnotation";

    public static void main(String[] args) throws Exception {
        Path root = repoRoot();
        Path wav = fileArg(args, 0, root.resolve("tests/test.wav"));
        Path onnx = fileArg(args, 1, root.resolve("docs/models/Just-Bird_fp32.onnx"));
        Path out = args.length > 2
                ? Paths.get(args[2])
                : root.resolve("tests/java/raven_selection_table.txt");

        float[] audio = loadWav(wav.toFile());

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        try (OrtSession session = env.createSession(onnx.toString())) {
            Map<String, String> meta = session.getMetadata().getCustomMetadata();
            float conf = floatArg(args, 3, meta, "default_conf", 0.18f);
            float gap = floatArg(args, 4, meta, "default_song_gap", 0.1f);
            Map<Integer, String> names = parseNames(meta.get("names"));

            try (OnnxTensor tAudio = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(audio), new long[] {audio.length});
                    OnnxTensor tConf = OnnxTensor.createTensor(
                            env, FloatBuffer.wrap(new float[] {conf}), new long[] {1});
                    OnnxTensor tGap = OnnxTensor.createTensor(
                            env, FloatBuffer.wrap(new float[] {gap}), new long[] {1})) {

                Map<String, OnnxTensor> feeds = new HashMap<String, OnnxTensor>();
                feeds.put("audio", tAudio);
                feeds.put("conf", tConf);
                feeds.put("song_gap", tGap);

                System.err.println("model     " + onnx);
                System.err.println("audio     " + wav + "  ("
                        + String.format(Locale.US, "%.1f", audio.length / (float) SAMPLE_RATE)
                        + " s)");
                System.err.println("conf      " + conf);
                System.err.println("song_gap  " + gap);
                System.err.println("running...");

                try (OrtSession.Result result = session.run(feeds)) {
                    String table = toRaven((OnnxTensor) result.get(0), names);
                    if (out.getParent() != null) {
                        Files.createDirectories(out.getParent());
                    }
                    Files.write(out, table.getBytes(StandardCharsets.UTF_8));
                    System.out.print(table);
                    System.err.println("wrote     " + out);
                }
            }
        }
    }

    static Path repoRoot() {
        Path dir = Paths.get("").toAbsolutePath();
        for (Path p = dir; p != null; p = p.getParent()) {
            if (Files.isDirectory(p.resolve("src/inference"))
                    && Files.isDirectory(p.resolve("tests"))) {
                return p;
            }
        }
        return dir;
    }

    static Path fileArg(String[] args, int i, Path fallback) {
        Path p = args.length > i ? Paths.get(args[i]) : fallback;
        if (!Files.isRegularFile(p)) {
            throw new IllegalArgumentException("Missing file: " + p);
        }
        return p.toAbsolutePath().normalize();
    }

    static float floatArg(
            String[] args, int i, Map<String, String> meta, String key, float fallback) {
        if (args.length > i) {
            return Float.parseFloat(args[i]);
        }
        String value = meta.get(key);
        if (value == null || value.length() == 0) {
            return fallback;
        }
        return Float.parseFloat(value);
    }

    static Map<Integer, String> parseNames(String json) {
        Map<Integer, String> names = new LinkedHashMap<Integer, String>();
        if (json == null) {
            return names;
        }
        Matcher matcher = Pattern.compile("\"(\\d+)\"\\s*:\\s*\"([^\"]*)\"").matcher(json);
        while (matcher.find()) {
            names.put(Integer.valueOf(matcher.group(1)), matcher.group(2));
        }
        return names;
    }

    static String toRaven(OnnxTensor detections, Map<Integer, String> names) throws Exception {
        long[] shape = detections.getInfo().getShape();
        int rows = shape.length == 0 ? 0 : (int) shape[0];
        int cols = shape.length > 1 ? (int) shape[1] : 8;
        StringBuilder table = new StringBuilder(RAVEN_HEADER).append('\n');
        if (rows == 0) {
            return table.toString();
        }

        FloatBuffer buffer = detections.getFloatBuffer();
        buffer.rewind();
        float[][] songs = new float[rows][cols];
        for (int r = 0; r < rows; r++) {
            buffer.get(songs[r]);
        }
        Arrays.sort(songs, Comparator.comparingDouble(row -> row[0]));

        for (int i = 0; i < rows; i++) {
            float[] row = songs[i];
            int classId = Math.round(row[7]);
            String species = names.containsKey(classId)
                    ? names.get(classId)
                    : String.valueOf(classId);
            table.append(i + 1)
                    .append("\tSpectrogram 1\t1\t")
                    .append(String.format(Locale.US, "%.1f", row[0]))
                    .append('\t')
                    .append(String.format(Locale.US, "%.1f", row[1]))
                    .append('\t')
                    .append(Math.round(row[2]))
                    .append('\t')
                    .append(Math.round(row[3]))
                    .append('\t')
                    .append(species)
                    .append('\n');
        }
        return table.toString();
    }

    static float[] loadWav(File file) throws Exception {
        AudioInputStream raw = AudioSystem.getAudioInputStream(file);
        AudioFormat src = raw.getFormat();
        AudioFormat pcm = new AudioFormat(
                AudioFormat.Encoding.PCM_SIGNED,
                src.getSampleRate(),
                16,
                src.getChannels(),
                src.getChannels() * 2,
                src.getSampleRate(),
                false);
        AudioInputStream in = src.matches(pcm) ? raw : AudioSystem.getAudioInputStream(pcm, raw);
        try {
            byte[] bytes = readAll(in);
            int channels = pcm.getChannels();
            int n = bytes.length / (2 * channels);
            float[] mono = new float[n];
            for (int i = 0; i < n; i++) {
                float sum = 0f;
                for (int c = 0; c < channels; c++) {
                    int j = (i * channels + c) * 2;
                    int sample = (bytes[j] & 0xff) | (bytes[j + 1] << 8);
                    if (sample >= 32768) {
                        sample -= 65536;
                    }
                    sum += sample / 32768f;
                }
                mono[i] = sum / channels;
            }
            int sr = Math.round(src.getSampleRate());
            if (sr != SAMPLE_RATE) {
                System.err.println("resampling " + sr + " Hz -> " + SAMPLE_RATE + " Hz");
                mono = resample(mono, sr, SAMPLE_RATE);
            }
            if (mono.length < SAMPLE_RATE * 3) {
                throw new IllegalArgumentException("Audio shorter than 3 seconds.");
            }
            return mono;
        } finally {
            in.close();
            if (in != raw) {
                raw.close();
            }
        }
    }

    static float[] resample(float[] x, int from, int to) {
        int n = (int) ((long) x.length * to / from);
        float[] y = new float[Math.max(n, 1)];
        for (int i = 0; i < n; i++) {
            double pos = (double) i * from / to;
            int j = (int) pos;
            float t = (float) (pos - j);
            float a = x[Math.min(j, x.length - 1)];
            float b = x[Math.min(j + 1, x.length - 1)];
            y[i] = a + t * (b - a);
        }
        return y;
    }

    static byte[] readAll(InputStream in) throws Exception {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[65536];
        int n;
        while ((n = in.read(buf)) >= 0) {
            out.write(buf, 0, n);
        }
        return out.toByteArray();
    }
}
