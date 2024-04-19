package net.mlcraft.elevationmod;

import com.mojang.serialization.Codec;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.util.dynamic.CodecHolder;
import net.minecraft.world.gen.densityfunction.DensityFunction;

import java.awt.*;
import java.io.*;
import java.util.ArrayList;

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

    private static short[] elevation = null;
    private static int length;

    public static void init() throws IOException {
        var file = FabricLoader.getInstance().getConfigDir().resolve("mlcraft/elevation").toFile();

        var d = new DataInputStream(new FileInputStream(file));

        elevation = new short[d.available()/Short.BYTES];

        for (int i = 0 ; i < elevation.length; i++){
            elevation[i] = d.readShort();
        }

        length = (int)Math.sqrt(elevation.length);
    }

    private static double getElevationAt(int x, int y){
        x = Math.floorMod(x, length);
        y = Math.floorMod(y, length);

        var v = elevation[x*length+y];

        return ((Short.toUnsignedInt(v) / 65535f) * 256) + 60;
    }

}
