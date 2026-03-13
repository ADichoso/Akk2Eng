from tqdm.auto import tqdm
import random
import model.decoder as decoder
import torch

def train_val_model(model, optimizer, criterion, scheduler, train_loader, val_loader, device, vocab, epochs, save_path):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_tokens = 0
        correct_tokens = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=False
        )

        #Training Phase
        for src, tgt in progress_bar:
            src = src.to(device)
            tgt = tgt.to(device)

            # Skip too-short sequences
            if tgt.size(1) < 2:
                continue

            # teacher forcing: show the model the ground truth in sequence so that it can examine the real previous word to predict the next one
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Clamp token indices safely
            tgt_input = torch.clamp(tgt_input, 0, len(vocab)-1)
            tgt_output = torch.clamp(tgt_output, 0, len(vocab)-1)

            # Skip batches with invalid token IDs
            if src.max() >= len(vocab) or src.min() < 0:
                print("Invalid src token in batch. Skipping.")
                continue
            if tgt_input.max() >= len(vocab) or tgt_input.min() < 0:
                print("Invalid tgt_input token in batch. Skipping.")
                continue
            if tgt_output.max() >= len(vocab) or tgt_output.min() < 0:
                print("Invalid tgt_output token in batch. Skipping.")
                continue

            optimizer.zero_grad()
            output = model(src, tgt_input)  # (batch, seq_len, vocab_size)

            # Flatten for loss calculation
            output_flat = output.reshape(-1, output.shape[-1])
            tgt_output_flat = tgt_output.reshape(-1)
            loss = criterion(output_flat, tgt_output_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            preds = output.argmax(-1)
            mask = tgt_output != vocab["<pad>"]
            correct_tokens += (preds == tgt_output).masked_select(mask).sum().item()
            total_tokens += mask.sum().item()

            # Update progress bar
            progress_bar.set_postfix(
                batch_loss=loss.item(),
                avg_loss=total_loss / (progress_bar.n + 1),
                batch_acc=correct_tokens / max(1, total_tokens)
            )

        train_loss = total_loss / len(train_loader)
        train_acc = correct_tokens / max(1, total_tokens)
        
        #Validation Phase
        model.eval()
        val_loss = 0
        val_tokens = 0
        val_correct = 0
        with torch.no_grad():
            for src, tgt in val_loader:  # your validation DataLoader
                src = src.to(device)
                tgt = tgt.to(device)

                if tgt.size(1) < 2:
                    continue

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                tgt_input = torch.clamp(tgt_input, 0, len(vocab)-1)
                tgt_output = torch.clamp(tgt_output, 0, len(vocab)-1)

                output = model(src, tgt_input)

                output_flat = output.reshape(-1, output.shape[-1])
                tgt_output_flat = tgt_output.reshape(-1)
                loss = criterion(output_flat, tgt_output_flat)
                val_loss += loss.item()

                preds = output.argmax(-1)
                mask = tgt_output != vocab["<pad>"]
                val_correct += (preds == tgt_output).masked_select(mask).sum().item()
                val_tokens += mask.sum().item()

        val_loss /= len(val_loader)
        val_acc = val_correct / max(1, val_tokens)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}")

        # Save model for this epoch
        torch.save({
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
            "accuracy": val_acc,
            "vocab": vocab
        }, f"{save_path}_epoch{epoch+1}_vAcc{val_acc:.4f}_vLoss{val_loss:.4f}.pth")

        #Sample translation
        model.eval()
        
        if epoch % 5 == 0:
            print("MBR MONITORING")
            # MBR sampling (only a small batch)
            with torch.no_grad():
                sample_batch = next(iter(val_loader))
                idx = random.randint(0, sample_batch[0].size(0)-1)

                sample_src = sample_batch[0][idx].unsqueeze(0).to(device)
                sample_tgt = sample_batch[1][idx]

                pred_tokens = decoder.mbr_decode(
                    model,
                    sample_src,
                    max_len=decoder.MAX_LEN, #Lower later on
                    sos_id=vocab["<sos>"],
                    eos_id=vocab["<eos>"],
                    samples=8
                )

                print("\n--- MBR SAMPLE ---")
                print("SRC :", decoder.decode_tokens(sample_src[0].cpu().tolist()))
                print("TGT :", decoder.decode_tokens(sample_tgt.tolist()))
                print("PRED:", decoder.decode_tokens(pred_tokens))
                print("------------------")

        print(f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")