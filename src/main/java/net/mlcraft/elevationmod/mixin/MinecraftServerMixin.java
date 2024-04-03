package net.mlcraft.elevationmod.mixin;

import net.mlcraft.elevationmod.MLCraft;
import net.mlcraft.elevationmod.world.gen.biome.source.MLCraftBiomeSource;
import net.mlcraft.elevationmod.world.gen.chunk.MLCraftChunkGenerator;
import net.minecraft.registry.*;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.WorldGenerationProgressListener;
import net.minecraft.world.dimension.DimensionOptions;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import java.util.Map;

@Mixin(MinecraftServer.class)
public abstract class MinecraftServerMixin {
    @Shadow
    @Final
    private CombinedDynamicRegistries<ServerDynamicRegistryType> combinedDynamicRegistries;


    @Inject(method = "createWorlds", at = @At("HEAD"))
    private void mlcraft_grabServer(WorldGenerationProgressListener worldGenerationProgressListener, CallbackInfo ci) {
        MinecraftServer server = (MinecraftServer) (Object) this;
        MLCraft.SERVER = server;
        Registry<DimensionOptions> registry = this.combinedDynamicRegistries.getCombinedRegistryManager().get(RegistryKeys.DIMENSION);
        for (Map.Entry<RegistryKey<DimensionOptions>, DimensionOptions> entry : registry.getEntrySet()) {
            if (entry.getValue().chunkGenerator().getBiomeSource() instanceof MLCraftBiomeSource abs) {
                try {
                    abs.findBiomeMap(server, entry.getKey().getValue().toString());
                } catch (Exception e) {
                    throw new IllegalStateException(e);
                }
            }
            if (entry.getValue().chunkGenerator() instanceof MLCraftChunkGenerator cg) {
                try {
                    cg.findMaps(server, entry.getKey().getValue().toString());
                } catch (Exception e) {
                    throw new IllegalStateException("error initializing: could not find maps for dimension "+entry.getKey().getValue().toString());
                }
            }
        }
    }
}