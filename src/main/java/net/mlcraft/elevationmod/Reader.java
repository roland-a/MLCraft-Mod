package net.mlcraft.elevationmod;

import com.mojang.serialization.Codec;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.util.dynamic.CodecHolder;
import net.minecraft.world.gen.densityfunction.DensityFunction;

import java.awt.*;
import java.io.IOException;

import static javax.imageio.ImageIO.*;

public class Reader {
    public static DensityFunction.Base MC_NOISE = new DensityFunction.Base(){
        private final CodecHolder<? extends DensityFunction> CODEC_HOLDER = CodecHolder.of(CODEC);

        @Override
        public double sample(NoisePos pos) {
            if (pos.blockY() < getElevationAt(pos.blockX(), pos.blockZ())) return 1;

            return -1;
        }

        @Override
        public double minValue() {
            return -1;
        }

        @Override
        public double maxValue() {
            return 1;
        }

        @Override
        public CodecHolder<? extends DensityFunction> getCodecHolder() {
            return CODEC_HOLDER;
        }
    };

    public static final Codec<? extends DensityFunction> CODEC = Codec.unit(MC_NOISE);

    public static double[][] elevation = null;

    public static void init() throws IOException {
        var image = read(FabricLoader.getInstance().getConfigDir().resolve("mlcraft/elevation.png").toFile());

        elevation = new double[image.getWidth()][image.getHeight()];

        for (int x = 0; x < image.getWidth(); x++){
            for (int y = 0; y < image.getHeight(); y++){
                int gray = new Color(image.getRGB(x,y)).getRed();

                elevation[x][y] = (gray / 255d) * 256 + 70;
            }
        }
    }

    private static double getElevationAt(int x, int y){
        x = Math.floorMod(x, elevation.length);
        y = Math.floorMod(y, elevation[0].length);

        return elevation[x][y];
    }

}
